"""训练/评估引擎：设备解析、数据准备、指标、训练与评估循环、检查点读写。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader

from .data import (MultimodalDataset, Scenario, build_exist_masks, fit_normalizers,
                   load_splits, prepare_split)
from .losses import (build_teacher_batch, classification_loss, distillation_loss,
                     local_context_target, reconstruction_loss, regression_loss, similarity_loss)
from .paths import MODALITIES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """auto：做一次真实 CUDA 算子自检，失败则回退 CPU。"""
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested.startswith("cuda"):
            raise RuntimeError("指定了 cuda，但当前环境没有可用 CUDA")
        return torch.device("cpu")
    try:
        probe = torch.relu(torch.ones(8, device="cuda:0")).sum().item()
        assert probe == 8.0
        return torch.device("cuda:0")
    except Exception as error:  # noqa: BLE001
        if requested.startswith("cuda"):
            raise
        print(f"CUDA 自检失败，回退 CPU：{error}", flush=True)
        return torch.device("cpu")


def concordance_correlation(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if truth.std() < 1e-8 or estimate.std() < 1e-8:
        return 0.0
    covariance = np.mean((truth - truth.mean()) * (estimate - estimate.mean()))
    denominator = truth.var() + estimate.var() + (truth.mean() - estimate.mean()) ** 2
    return float(2 * covariance / denominator) if denominator > 1e-12 else 0.0


def compute_metrics(labels: np.ndarray, predictions: np.ndarray,
                    targets: np.ndarray, estimates: np.ndarray) -> dict:
    pearson = float(np.corrcoef(targets, estimates)[0, 1]) if np.std(estimates) > 1e-8 else 0.0
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "mae": float(mean_absolute_error(targets, estimates)),
        "pearson": pearson,
        "ccc": concordance_correlation(targets, estimates),
        "prediction_distribution": np.bincount(predictions, minlength=3).tolist(),
    }


def subset_prepared(prepared: dict, count: int) -> dict:
    """仅取前 count 个样本（冒烟测试用）；返回视图，不复制数据。"""
    if not count or count >= len(prepared["class"]):
        return prepared
    return {"features": {name: values[:count] for name, values in prepared["features"].items()},
            "masks": {name: {key: value[:count] for key, value in masks.items()}
                      for name, masks in prepared["masks"].items()},
            "class": prepared["class"][:count], "regression": prepared["regression"][:count],
            "ids": prepared["ids"][:count]}


def load_prepared_splits(path: Path, need_test: bool = True, max_train: int = 0,
                         max_valid: int = 0) -> dict:
    """按 train→valid→test 顺序逐个归一化并释放原始数组，控制内存峰值。"""
    splits = load_splits(path)
    exist_train = build_exist_masks(splits["train"])
    normalizers = fit_normalizers(splits["train"], exist_train)
    prepared = {}
    prepared["train"] = subset_prepared(prepare_split(splits.pop("train"), normalizers, exist_train),
                                        max_train)
    prepared["valid"] = subset_prepared(prepare_split(splits.pop("valid"), normalizers), max_valid)
    if need_test and "test" in splits:
        prepared["test"] = prepare_split(splits.pop("test"), normalizers)
    splits.clear()
    return {"prepared": prepared, "normalizers": normalizers}


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def make_loader(prepared: dict, batch_size: int, seed: int, train_mode: bool,
                scenario: Scenario | None = None, with_clean: bool = True, shuffle: bool = False):
    dataset = MultimodalDataset(prepared, scenario=scenario, seed=seed, with_clean=with_clean,
                                train_mode=train_mode)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def reconstruction_targets(batch: dict, output: dict, model, teacher_tokens: dict | None,
                           settings: dict) -> dict:
    """重建目标（均在隐空间）：self=干净输入的自编码，teacher=教师编码，context=邻域观测均值。"""
    mode = settings.get("rec_target", "self")
    targets, hidden = {}, None
    if output.get("rec"):
        hidden = output["rec"][MODALITIES[0]].shape[-1]
    for name in MODALITIES:
        if mode == "teacher" and teacher_tokens is not None and teacher_tokens[name].shape[-1] == hidden:
            targets[name] = teacher_tokens[name]
        elif mode == "self" and hasattr(model, "encode_clean") and f"{name}_clean" in batch:
            if "self_targets" not in targets:
                targets["self_targets"] = model.encode_clean(batch)
            targets[name] = targets["self_targets"][name]
        else:
            tokens = output.get("tokens_encoded", output.get("tokens", {}))
            targets[name] = local_context_target(tokens[name], batch[f"{name}_observed"],
                                                 int(settings.get("ctx_span", 6)))
    targets.pop("self_targets", None)
    return targets


def run_epoch(model, loader, optimizer, device, settings: dict, teacher=None, train: bool = True) -> dict:
    """单个 epoch 的训练或评估；返回指标与分项损失。"""
    model.train(train)
    if teacher is not None:
        teacher.eval()
    totals = {key: 0.0 for key in ("loss", "cls", "reg", "rec", "distill")}
    count, labels, predictions, targets, estimates = 0, [], [], [], []
    for raw in loader:
        batch = move_batch(raw, device)
        n = batch["cls"].shape[0]
        teacher_output, teacher_tokens = None, None
        if teacher is not None:
            clean_batch = build_teacher_batch(batch)
            with torch.no_grad():
                tokens, _ = teacher.encode(clean_batch)
                teacher_tokens = tokens
                teacher_output = teacher(clean_batch)
        with torch.set_grad_enabled(train):
            output = model(batch)
            loss_cls = classification_loss(output["logits"], batch["cls"], settings.get("class_weights"))
            loss_reg = regression_loss(output["intensity"], batch["reg"])
            loss = loss_cls + settings["reg_weight"] * loss_reg
            loss_rec = torch.zeros((), device=device)
            if settings["rec_weight"] > 0 and output.get("rec"):
                rec_targets = reconstruction_targets(batch, output, model, teacher_tokens, settings)
                if rec_targets:
                    loss_rec = reconstruction_loss(output["rec"], output["rec_valid"], rec_targets)
                    loss = loss + settings["rec_weight"] * loss_rec
            loss_distill = torch.zeros((), device=device)
            if teacher_output is not None and settings["distill_weight"] > 0:
                kl, mse = distillation_loss(output["logits"], teacher_output["logits"],
                                            output["intensity"], teacher_output["intensity"],
                                            settings.get("temperature", 2.0))
                loss_distill = kl + mse
                loss = loss + settings["distill_weight"] * loss_distill
            if settings.get("similarity_weight", 0.0) > 0 and "invariant" in output:
                loss = loss + settings["similarity_weight"] * similarity_loss(output["invariant"],
                                                                            output["specific"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get("clip", 1.0))
                optimizer.step()
        count += n
        totals["loss"] += float(loss) * n
        totals["cls"] += float(loss_cls) * n
        totals["reg"] += float(loss_reg) * n
        totals["rec"] += float(loss_rec) * n
        totals["distill"] += float(loss_distill) * n
        labels.extend(batch["cls"].detach().cpu().numpy())
        predictions.extend(output["logits"].argmax(dim=-1).detach().cpu().numpy())
        targets.extend(batch["reg"].detach().cpu().numpy())
        estimates.extend(output["intensity"].detach().cpu().numpy())
    metrics = compute_metrics(np.asarray(labels), np.asarray(predictions),
                              np.asarray(targets), np.asarray(estimates))
    metrics.update({key: value / max(1, count) for key, value in totals.items()})
    return metrics


def save_checkpoint(path: Path, model, model_name: str, model_kwargs: dict, normalizers: dict,
                    metrics: dict, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "model_name": model_name, "model_kwargs": model_kwargs,
               "normalizers": normalizers, "metrics": metrics}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device):
    """读取检查点并重建模型（供评估、缺失分析与附件3 推理复用）。"""
    from .models import build_model  # 局部导入避免循环依赖

    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(payload["model_name"], **payload.get("model_kwargs", {})).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload

