"""数据层：加载附件2 特征、构建双掩码（padding / missing）、缺失模拟与标准化。

双掩码定义（与赛题“局部时段模态缺失”语义对齐）：
    exist(t)    位置 t 是否位于该样本的有效时间轴内（长度边界内）——由显式 *_mask / *_lengths 或非零段确定
    observed(t) exist(t) 且该位置有真实观测（特征非全零）
    missing(t)  exist(t) 且无观测 —— 即“本该存在但丢失”的局部连续区间

附件2 对齐版有效段内部没有零（零值全部来自长度填充），因此训练时对 observed 位置人为制造连续缺失；
附件3 的缺失是真实零段，推理时用 detect_missing 直接检测，保证训练/推理语义一致。
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .paths import MODALITIES

EPS = 1e-8


# ---------------------------------------------------------------- 掩码与标准化


def row_nonzero(values: np.ndarray) -> np.ndarray:
    """逐时间位置判断是否存在真实观测。"""
    return np.linalg.norm(values, axis=-1) > 1e-8


def span_from_nonzero(nonzero: np.ndarray) -> np.ndarray:
    """由非零行推出“有效时间轴”的连续区间（含首尾），边界之外视为长度填充。"""
    exist = np.zeros_like(nonzero, dtype=bool)
    any_row = nonzero.any(axis=1)
    if not any_row.any():
        return exist
    first = nonzero.argmax(axis=1)
    last = nonzero.shape[1] - 1 - nonzero[:, ::-1].argmax(axis=1)
    index = np.arange(nonzero.shape[1])[None, :]
    return any_row[:, None] & (index >= first[:, None]) & (index <= last[:, None])


def build_exist_masks(split: dict) -> dict:
    """优先使用显式掩码/长度，其次由非零段推断；返回 {modality: bool 数组 (N,T)}。"""
    text_tokens = np.asarray(split["text_bert"])
    exist = {"text": text_tokens[:, 1, :] != 0}
    for name in ("audio", "vision"):
        values = np.asarray(split[name])
        if f"{name}_mask" in split:
            mask = np.asarray(split[f"{name}_mask"], dtype=bool)
            exist[name] = mask if mask.shape == values.shape[:2] else span_from_nonzero(row_nonzero(values))
        elif f"{name}_lengths" in split:
            lengths = np.asarray(split[f"{name}_lengths"], dtype=np.int64)
            index = np.arange(values.shape[1])[None, :]
            exist[name] = index < lengths[:, None]
        else:
            exist[name] = span_from_nonzero(row_nonzero(values))
    return exist


def masked_mean_std(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按有效位置统计每维均值/标准差。"""
    flat = np.asarray(values, dtype=np.float32)
    valid = mask.astype(bool).reshape(-1)
    if not valid.any():
        return np.zeros(flat.shape[-1], dtype=np.float32), np.ones(flat.shape[-1], dtype=np.float32)
    selected = flat.reshape(-1, flat.shape[-1])[valid]
    mean = selected.mean(axis=0).astype(np.float32)
    std = np.maximum(selected.std(axis=0), 1e-4).astype(np.float32)
    return mean, std


def fit_normalizers(train_split: dict, exist: dict) -> dict:
    """训练集统计量：均值/方差只取“有观测”的位置。"""
    normalizers = {}
    for name in MODALITIES:
        values = np.asarray(train_split[name], dtype=np.float32)
        observed = exist[name] & row_nonzero(values)
        normalizers[name] = masked_mean_std(values, observed)
        del values
    return normalizers


def prepare_split(split: dict, normalizers: dict, exist: dict | None = None) -> dict:
    """归一化（clip ±10）并输出 float32 特征与三类掩码；原地运算以控制峰值内存。"""
    if exist is None:
        exist = build_exist_masks(split)
    features, masks = {}, {}
    for name in MODALITIES:
        values = np.asarray(split[name], dtype=np.float32)
        if not np.isfinite(values).all():
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        exist_mask = exist[name]
        # 关键：observed 必须在标准化之前用原始特征判定，否则标准化会把全零行变成非零（-mean/std）。
        observed = exist_mask & row_nonzero(values)
        mean, std = normalizers[name]
        values -= mean
        values /= std
        np.clip(values, -10.0, 10.0, out=values)
        values *= exist_mask[..., None]          # 长度填充位置强制为零，避免污染
        features[name] = values
        masks[name] = {"exist": exist_mask, "observed": observed}
        del values
    labels_class = np.asarray(split["classification_labels"], dtype=np.int64)
    if not np.isin(labels_class, [0, 1, 2]).all():
        raise ValueError("分类标签必须是 0/1/2")
    return {"features": features, "masks": masks, "class": labels_class,
            "regression": np.asarray(split["regression_labels"], dtype=np.float32),
            "ids": list(split["id"])}


def load_splits(path: Path) -> dict:
    """读取 pkl。内存提示：aligned 约 0.95 GB、resampled 约 0.85 GB 常驻。"""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------- 缺失模拟


@dataclass
class Scenario:
    """一个缺失场景：作用于哪些模态、落在何处、缺失多少。"""

    mode: str = "random"            # random / text / audio / vision / 'text+audio' 等组合
    position: str = "random"        # random / start / middle / end
    ratio: float = 0.3              # 相对该模态有效长度的缺失比例
    min_len: int = 3                # 单个连续缺失区间的最小窗数
    max_intervals: int = 3
    duration: int | None = None     # 指定单区间窗数（与 ratio 二选一）

    def modalities(self, rng: random.Random | None = None) -> tuple[str, ...]:
        chooser = rng or random
        if self.mode == "none":
            return ()
        if self.mode == "random":
            count = chooser.choice([1, 1, 2, 2, 3])
            return tuple(chooser.sample(list(MODALITIES), count))
        return tuple(part for part in self.mode.split("+") if part in MODALITIES)

    def to_dict(self) -> dict:
        return {"mode": self.mode, "position": self.position, "ratio": self.ratio,
                "min_len": self.min_len, "max_intervals": self.max_intervals, "duration": self.duration}


def mask_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """把布尔掩码展开成 [start, end) 连续区间列表。"""
    intervals, start = [], None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            intervals.append((start, index))
            start = None
    if start is not None:
        intervals.append((start, len(mask)))
    return intervals


def sample_scenario(rng: random.Random) -> Scenario:
    """训练时随机采样缺失场景：单/双/三模态、随机位置、10%–50% 缺失率。"""
    return Scenario(mode="random", position=rng.choice(["random", "start", "middle", "end"]),
                    ratio=round(rng.uniform(0.1, 0.5), 3))


def _place_segments(span: np.ndarray, scenario: Scenario, rng: random.Random) -> np.ndarray:
    """在有效段内部给出要置零的位置掩码（保证至少保留一个观测位置）。"""
    missing = np.zeros_like(span, dtype=bool)
    blocks = mask_intervals(span)
    if not blocks:
        return missing
    start, end = blocks[0] if scenario.position != "random" else rng.choice(blocks)
    length = end - start
    if length <= 1:                      # 只有 1 个有效位置时不做缺失，避免出现“整模态无观测”
        return missing
    if scenario.duration is not None:
        total, count = int(scenario.duration), 1
    else:
        total = int(round(scenario.ratio * length))
        count = max(1, min(scenario.max_intervals, rng.randint(1, scenario.max_intervals)))
    upper = max(1, length - 1)
    total = int(np.clip(total, min(scenario.min_len, upper), upper))
    if scenario.position in ("start", "middle", "end") or total <= scenario.min_len:
        count = 1
    if count == 1:
        if scenario.position == "start":
            first = start
        elif scenario.position == "end":
            first = end - total
        elif scenario.position == "middle":
            first = start + (length - total) // 2
        else:
            first = rng.randint(start, end - total)
        missing[first:first + total] = True
        return missing
    cursor, remaining = start, total
    for index in range(count):
        left = count - index
        room = end - cursor - scenario.min_len * (left - 1)
        if room <= 0:
            break
        piece = min(rng.randint(scenario.min_len, max(scenario.min_len, room)), room,
                    remaining - scenario.min_len * (left - 1) if left > 1 else remaining)
        piece = max(1, piece)
        missing[cursor:cursor + piece] = True
        cursor += piece
        remaining -= piece
    return missing


def simulate_missing(features: dict, masks: dict, scenario: Scenario, rng: random.Random):
    """按场景在有效段内制造连续缺失；返回被破坏的特征副本、三类掩码与区间记录。"""
    corrupted = {name: values.copy() for name, values in features.items()}
    observed = {name: masks[name]["observed"].copy() for name in MODALITIES}
    missing = {name: np.zeros_like(observed[name], dtype=bool) for name in MODALITIES}
    records = []
    for name in scenario.modalities(rng):
        segment = _place_segments(masks[name]["exist"], scenario, rng)
        if not segment.any():
            continue
        corrupted[name][segment] = 0.0
        observed[name] &= ~segment
        missing[name] |= segment
        for first, last in mask_intervals(segment):
            records.append({"modality": name, "start": int(first), "end": int(last - 1),
                            "length": int(last - first)})
    # 统一保证：特征非零 ⟺ observed（数据自带的段内零段也一并归零，避免标准化后的 −mean/std 泄漏）
    for name in MODALITIES:
        corrupted[name][~observed[name]] = 0.0
    return corrupted, observed, missing, records


def detect_missing(features: dict, masks: dict) -> dict:
    """附件3 推理用：把有效段内的全零段判定为缺失（只检测，不改变特征）。"""
    return {name: masks[name]["exist"] & ~row_nonzero(features[name]) for name in MODALITIES}


# ---------------------------------------------------------------- Dataset


def _as_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array))


class MultimodalDataset(Dataset):
    """带双掩码的样本集；训练时每个 epoch 重新随机缺失，评估时使用固定场景。"""

    def __init__(self, prepared: dict, scenario: Scenario | None = None, seed: int = 0,
                 with_clean: bool = True, train_mode: bool = False):
        self.features = prepared["features"]
        self.masks = prepared["masks"]
        self.classes = prepared["class"]
        self.regressions = prepared["regression"]
        self.ids = prepared["ids"]
        self.scenario = scenario
        self.seed = seed
        self.with_clean = with_clean
        self.train_mode = train_mode
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.classes)

    def reseed(self, epoch: int) -> None:
        """训练时每个 epoch 重新采样缺失模式。"""
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict:
        features = {name: self.features[name][index] for name in MODALITIES}
        masks = {name: {"exist": self.masks[name]["exist"][index],
                        "observed": self.masks[name]["observed"][index]} for name in MODALITIES}
        rng = random.Random((self.seed * 1000003) + index + (self.epoch * 7919 if self.train_mode else 0))
        scenario = sample_scenario(rng) if self.train_mode else (self.scenario or Scenario())
        corrupted, observed, missing, _ = simulate_missing(features, masks, scenario, rng)
        sample = {}
        for name in MODALITIES:
            sample[name] = _as_tensor(corrupted[name])
            sample[f"{name}_exist"] = _as_tensor(masks[name]["exist"].astype(np.float32))
            sample[f"{name}_observed"] = _as_tensor(observed[name].astype(np.float32))
            sample[f"{name}_missing"] = _as_tensor(missing[name].astype(np.float32))
            if self.with_clean:
                sample[f"{name}_clean"] = _as_tensor(features[name])
        sample["cls"] = torch.tensor(int(self.classes[index]), dtype=torch.long)
        sample["reg"] = torch.tensor(float(self.regressions[index]), dtype=torch.float32)
        sample["index"] = torch.tensor(index, dtype=torch.long)
        return sample
