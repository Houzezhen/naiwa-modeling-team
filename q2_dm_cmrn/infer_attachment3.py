"""附件3 推理：自动检测有效段内的连续缺失区间，输出极性、强度、概率与缺失统计。

示例：
    python infer_attachment3.py --checkpoint outputs/dm_cmrn/best.pt --device cpu
    python infer_attachment3.py --checkpoint outputs/dm_cmrn/best.pt --text-mode distilbert --device cuda:0

说明：
- 对齐版本提供 text_bert（token id）而没有 768 维文本特征，因此用 DistilBERT 最后 4 层均值重建；
  未对齐版本提供 raw_text，同样用 DistilBERT 重建；`--text-mode zero` 可做“文本全缺失”对照。
- 未对齐版本的语音/视觉是 500 步定长 padding，按长度感知等时长重采样到 50 窗（与 q2_baseline 一致）。
- 缺失掩码由“有效段内的全零位置”自动检测，与训练时的缺失语义保持一致。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.data import detect_missing, mask_intervals, row_nonzero, span_from_nonzero  # noqa: E402
from src.engine import load_checkpoint, resolve_device, set_seed  # noqa: E402
from src.paths import ATTACHMENT3_DIR, CLASS_NAMES, MODALITIES, OUTPUT_ROOT  # noqa: E402

TEXT_MODEL = "distilbert-base-uncased"
TEXT_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


def bin_to_steps(values: np.ndarray, length: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """长度感知的等时长加权分箱：把 (T,d) 压到 (steps,d)，并给出有效位置掩码。"""
    values = np.asarray(values, dtype=np.float32)
    length = int(min(max(length, 0), values.shape[0]))
    result = np.zeros((steps, values.shape[-1]), dtype=np.float32)
    mask = np.zeros(steps, dtype=bool)
    if length == 0:
        return result, mask
    edges = np.linspace(0.0, float(length), steps + 1)
    for index in range(steps):
        start, end = edges[index], edges[index + 1]
        first, last = int(np.floor(start)), min(length, int(np.ceil(end)))
        if last <= first:
            first, last = min(first, length - 1), min(first, length - 1) + 1
        frames = np.arange(first, last)
        overlap = np.maximum(0.0, np.minimum(frames + 1.0, end) - np.maximum(frames, start))
        total = float(overlap.sum())
        if total > 0:
            result[index] = (values[frames] * overlap[:, None]).sum(axis=0) / total
            mask[index] = True
    return result, mask


def first_sample(array) -> np.ndarray:
    """附件3 的字段常带前置批次维（形状 (1,T,d)），这里统一去掉。"""
    values = np.asarray(array)
    return values[0] if values.ndim == 3 else values


def build_text(sample: dict, text_mode: str, tokenizer, text_model, steps: int, device) -> tuple:
    """返回 (features (steps,768), exist_mask, 来源说明)。"""
    if text_mode == "distilbert" and tokenizer is not None:
        if "text_bert" in sample:
            token_data = first_sample(sample["text_bert"])
            input_ids = torch.from_numpy(token_data[0, :steps].astype(np.int64))[None].to(device)
            attention = torch.from_numpy((token_data[1, :steps] != 0).astype(np.int64))[None].to(device)
            source = "provided_text_bert_token_ids"
        else:
            raw = str(np.asarray(sample["raw_text"]).reshape(-1)[0])
            encoded = tokenizer([raw], padding="max_length", truncation=True, max_length=steps,
                                return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention = encoded["attention_mask"].to(device)
            source = "tokenized_raw_text"
        with torch.no_grad():
            output = text_model(input_ids=input_ids, attention_mask=attention,
                                output_hidden_states=True, return_dict=True)
            hidden = torch.stack(output.hidden_states[-4:], dim=0).mean(dim=0)[0].cpu().numpy()
        features = np.zeros((steps, hidden.shape[-1]), dtype=np.float32)
        exist = np.zeros(steps, dtype=bool)
        count = min(hidden.shape[0], steps)
        features[:count] = hidden[:count]
        exist[:count] = attention[0, :count].cpu().numpy().astype(bool)
        return features, exist, source
    return np.zeros((steps, 768), dtype=np.float32), np.zeros(steps, dtype=bool), "explicit_zero_text"


def prepare_sample(sample: dict, variant: str, normalizers: dict, text_mode: str, tokenizer, text_model,
                   steps: int, device, exist_mode: str = "full") -> dict:
    """把单条附件3 样本整理成模型输入（特征 + 三类掩码）。

    exist_mode 决定“长度边界外的零”如何解读（附件3 无长度字段，两种解读都需报告）：
        span：边界外的零视为长度填充（与附件2 训练口径一致），缺失只统计有效段内部
        full：50 窗（未对齐为 500 步画布）内的零都视为“本该存在但缺失”，从而报告缺失区间
    """
    text_values, text_exist, text_source = build_text(sample, text_mode, tokenizer, text_model, steps, device)
    features, exist = {"text": text_values}, {"text": text_exist}
    for name in ("audio", "vision"):
        raw = first_sample(sample[name]).astype(np.float32)
        if raw.shape[0] == steps:                     # 已是 50 窗（对齐版本）
            values = raw
            if exist_mode == "full":
                mask = np.arange(steps) > 0           # 该数据集第 0 窗恒为占位，其余按“应存在”处理
            elif f"{name}_mask" in sample:
                mask = np.asarray(sample[f"{name}_mask"], dtype=bool).reshape(-1)[:steps]
            elif f"{name}_lengths" in sample:
                mask = np.arange(steps) < int(np.asarray(sample[f"{name}_lengths"]).reshape(-1)[0])
            else:
                mask = span_from_nonzero(row_nonzero(values)[None])[0]
        else:                                         # 未对齐版本：500 步定长画布
            if exist_mode == "full":
                length = raw.shape[0]
            elif f"{name}_lengths" in sample:
                length = int(np.asarray(sample[f"{name}_lengths"]).reshape(-1)[0])
            else:
                nonzero_rows = np.flatnonzero(row_nonzero(raw))
                length = int(nonzero_rows[-1] + 1) if len(nonzero_rows) else 0
            values, mask = bin_to_steps(raw, length, steps)
            if exist_mode == "full":
                mask = np.ones(steps, dtype=bool)
        features[name], exist[name] = values, mask
    normalized, observed = {}, {}
    for name in MODALITIES:
        values = np.asarray(features[name], dtype=np.float32)
        observed[name] = exist[name] & row_nonzero(values)      # 标准化前用原始特征判定观测
        mean, std = normalizers[name]
        values -= mean
        values /= std
        np.clip(values, -10.0, 10.0, out=values)
        values *= exist[name][..., None]
        values[~observed[name]] = 0.0                           # 特征非零 ⟺ observed
        normalized[name] = values
    masks = {name: {"exist": exist[name], "observed": observed[name]} for name in MODALITIES}
    return {"features": normalized, "masks": masks, "missing": detect_missing(normalized, masks),
            "text_source": text_source}


def to_batch(prepared: dict, device) -> dict:
    batch = {}
    for name in MODALITIES:
        batch[name] = torch.from_numpy(prepared["features"][name])[None].to(device)
        batch[f"{name}_exist"] = torch.from_numpy(
            prepared["masks"][name]["exist"].astype(np.float32))[None].to(device)
        batch[f"{name}_observed"] = torch.from_numpy(
            prepared["masks"][name]["observed"].astype(np.float32))[None].to(device)
        batch[f"{name}_missing"] = torch.from_numpy(
            prepared["missing"][name].astype(np.float32))[None].to(device)
    return batch


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--attachment3", type=Path, default=ATTACHMENT3_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "attachment3")
    parser.add_argument("--text-mode", choices=("distilbert", "zero"), default="distilbert")
    parser.add_argument("--exist-mode", choices=("span", "full"), default="full",
                        help="span=边界外的零视为长度填充；full=50 窗/500 步画布内的零视为缺失")
    parser.add_argument("--variants", nargs="+", choices=("aligned", "unaligned"), default=["aligned", "unaligned"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    normalizers = payload["normalizers"]
    tokenizer = text_model = None
    if args.text_mode == "distilbert":
        try:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION, use_fast=True)
            text_model = AutoModel.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION).to(device).eval()
        except Exception as error:  # noqa: BLE001
            raise RuntimeError("DistilBERT 不可用（需安装 transformers 且能访问模型仓库；"
                               "可先设置 HF_ENDPOINT=https://hf-mirror.com，或改用 --text-mode zero）") from error
    print(f"模型={payload['model_name']} 数据={args.attachment3} text_mode={args.text_mode} device={device}",
          flush=True)

    rows = []
    for variant in args.variants:
        folder = args.attachment3 / ("对齐版本" if variant == "aligned" else "未对齐版本")
        files = sorted(folder.glob("*.pkl"))
        if not files:
            print(f"跳过（目录为空）：{folder}", flush=True)
            continue
        for path in files:
            import pickle
            sample = pickle.loads(path.read_bytes())["test"]
            prepared = prepare_sample(sample, variant, normalizers, args.text_mode, tokenizer, text_model,
                                      args.steps, device, args.exist_mode)
            batch = to_batch(prepared, device)
            with torch.no_grad():
                output = model(batch)
                probabilities = torch.softmax(output["logits"], dim=-1)[0].cpu().numpy()
            record = {"variant": variant, "text_mode": args.text_mode, "file": path.name, "id": path.stem,
                      "text_source": prepared["text_source"],
                      "predicted_class": CLASS_NAMES[int(probabilities.argmax())],
                      "predicted_strength": float(output["intensity"][0].cpu()),
                      "prob_negative": float(probabilities[0]), "prob_neutral": float(probabilities[1]),
                      "prob_positive": float(probabilities[2])}
            for index, name in enumerate(MODALITIES):
                record[f"observed_{name}_bins"] = int(prepared["masks"][name]["observed"].sum())
                record[f"missing_{name}_bins"] = int(prepared["missing"][name].sum())
                intervals = mask_intervals(prepared["missing"][name])
                record[f"missing_{name}_intervals"] = ";".join(f"{a}-{b - 1}" for a, b in intervals)
                record[f"weight_{name}"] = float(output["weights"][0, index].cpu())
            rows.append(record)
            print(f"推理完成 {variant} {path.name} → {record['predicted_class']} "
                  f"({record['predicted_strength']:+.3f})", flush=True)

    if not rows:
        raise SystemExit("没有可推理的附件3 样本")
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "attachment3_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {"checkpoint": str(args.checkpoint), "model": payload["model_name"], "text_mode": args.text_mode,
               "exist_mode": args.exist_mode, "samples": len(rows), "metrics": None,
               "metrics_note": "附件3 无真实标签，不能计算 Accuracy/Macro-F1/MAE/Pearson，只报告预测与缺失统计。",
               "class_counts": {variant: frame[frame["variant"] == variant]["predicted_class"].value_counts().to_dict()
                                for variant in args.variants},
               "mean_strength": {variant: float(frame[frame["variant"] == variant]["predicted_strength"].mean())
                                 for variant in args.variants},
               "mean_missing_bins": {variant: {name: float(frame[frame["variant"] == variant][
                   f"missing_{name}_bins"].mean()) for name in MODALITIES} for variant in args.variants},
               "mean_weight": {variant: {name: float(frame[frame["variant"] == variant][
                   f"weight_{name}"].mean()) for name in MODALITIES} for variant in args.variants},
               "output_csv": str(args.output / "attachment3_predictions.csv")}
    (args.output / "attachment3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                          encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


