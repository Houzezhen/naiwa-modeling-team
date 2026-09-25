"""损失函数：分类、回归、局部重建、知识蒸馏，以及 MISA 的表示约束。"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .paths import MODALITIES


def classification_loss(logits: torch.Tensor, target: torch.Tensor,
                        class_weights: torch.Tensor | None = None) -> torch.Tensor:
    return F.cross_entropy(logits, target, weight=class_weights)


def regression_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target)


def local_context_target(tokens: torch.Tensor, observed: torch.Tensor, window: int) -> torch.Tensor:
    """缺失位置的“上下文均值”目标：前后 window 个观测 token 的加权平均（自监督修复目标）。"""
    hidden = tokens.shape[-1]
    weights = observed.unsqueeze(-1)
    kernel = 2 * window + 1
    ones = torch.ones(hidden, 1, kernel, device=tokens.device, dtype=tokens.dtype)
    numerator = F.conv1d((tokens * weights).transpose(1, 2), ones, padding=window, groups=hidden)
    denominator = F.conv1d(weights.transpose(1, 2), torch.ones(1, 1, kernel, device=tokens.device,
                                                              dtype=tokens.dtype), padding=window)
    return (numerator / denominator.clamp_min(1e-6)).transpose(1, 2)


def reconstruction_loss(reconstruction: dict, valid: dict, targets: dict,
                        weight_map: dict | None = None) -> torch.Tensor:
    """仅在“有缺失且重建有效”的位置计算 MSE。"""
    total, count = reconstruction[list(MODALITIES)[0]].new_zeros(()), 0
    for name in MODALITIES:
        if name not in reconstruction or name not in valid:
            continue
        mask = valid[name]
        if mask.sum() == 0:
            continue
        error = (reconstruction[name] - targets[name]) ** 2
        error = error.mean(dim=-1)                      # (B, T)
        weight = 1.0 if weight_map is None else float(weight_map.get(name, 1.0))
        total = total + weight * (error * mask).sum() / mask.sum()
        count += 1
    return total / max(1, count)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      student_regression: torch.Tensor, teacher_regression: torch.Tensor,
                      temperature: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    """分类用温度缩放 KL，回归用 MSE。"""
    log_softmax = F.log_softmax(student_logits / temperature, dim=-1)
    soft = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(log_softmax, soft, reduction="batchmean") * (temperature ** 2)
    mse = F.mse_loss(student_regression, teacher_regression)
    return kl, mse


def similarity_loss(invariant: dict, specific: dict) -> torch.Tensor:
    """MISA 风格约束：不变表示与特异表示尽量不相关（余弦相似度平方）。"""
    losses = []
    for name in MODALITIES:
        if name not in invariant or name not in specific:
            continue
        first = F.normalize(invariant[name], dim=-1)
        second = F.normalize(specific[name], dim=-1)
        losses.append((first * second).sum(dim=-1).pow(2).mean())
    return torch.stack(losses).mean() if losses else torch.zeros((), device=next(iter(invariant.values())).device)


def build_teacher_batch(batch: dict) -> dict:
    """把“干净输入 + 完整有效轴”组装成 Teacher 前向所需的 batch（用于生成重建目标）。"""
    clean = {}
    zeros = None
    for name in MODALITIES:
        clean[name] = batch[f"{name}_clean"]
        clean[f"{name}_exist"] = batch[f"{name}_exist"]
        clean[f"{name}_observed"] = batch[f"{name}_exist"]      # 干净输入下有效轴全部有观测
        clean[f"{name}_missing"] = torch.zeros_like(batch[f"{name}_exist"])
    return clean
