"""模型层：模态编码器、4 个 Baseline、DM-CMRN（双掩码 + 局部连续重建 + 质量门控融合）。

统一接口：forward(batch, teacher_features=None) -> dict
    batch 需包含 {m}, {m}_exist, {m}_observed, {m}_missing（m ∈ text/audio/vision），
    可选 {m}_clean（用于自重建目标）。
返回值：
    logits (B,3)、intensity (B,)、weights (B,3) 模态门控权重、ratios (B,3) 观测比例、
    rec（可选，各模态重建特征字典）、rec_conf（可选，各模态重建置信度）、pooled（各模态池化表示）
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .paths import MODALITIES, INPUT_DIMS, CLASS_NAMES


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """按掩码对时间维求均值；mask 形状 (B,T)，tokens 形状 (B,T,H)。"""
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    total = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / total


def status_ids(observed: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
    """双掩码状态编码：0=长度填充，1=有观测，2=本该存在但缺失。"""
    return torch.where(missing > 0, torch.full_like(observed, 2, dtype=torch.long),
                       torch.where(observed > 0, torch.ones_like(observed, dtype=torch.long),
                                   torch.zeros_like(observed, dtype=torch.long)))


class ModalEncoder(nn.Module):
    """单模态时序编码器：线性投影 + 深度可分离卷积 + 双掩码状态嵌入 + Transformer。"""

    def __init__(self, dim: int, hidden: int, layers: int = 2, nhead: int = 4,
                 dropout: float = 0.2, conv_kernel: int = 3, use_status_embed: bool = True):
        super().__init__()
        self.use_status_embed = use_status_embed
        self.projection = nn.Linear(dim, hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=conv_kernel, padding=conv_kernel // 2,
                              groups=max(1, hidden // 8))
        self.status = nn.Embedding(3, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
                                           dropout=dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, values: torch.Tensor, exist: torch.Tensor, observed: torch.Tensor,
                missing: torch.Tensor):
        hidden = self.projection(values)
        hidden = F.gelu(self.conv(hidden.transpose(1, 2)).transpose(1, 2))
        if self.use_status_embed:
            hidden = hidden + self.status(status_ids(observed, missing))
        hidden = self.input_norm(hidden)
        # 缺失位置与填充位置都不参与自注意力；两种情况需要回退，避免整行全屏蔽产生 NaN：
        #   (a) 该样本该模态有观测 → 用 observed；(b) 没有观测但有有效轴 → 用 exist；
        #   (c) 连有效轴都没有（整模态缺失）→ 注意力用全 1 掩码，但输出与池化仍按 0 处理。
        have_observation = observed.sum(dim=1, keepdim=True) > 0
        effective = torch.where(have_observation, observed > 0, exist > 0)
        attention_mask = torch.where(effective.any(dim=1, keepdim=True), effective,
                                     torch.ones_like(effective))
        encoded = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        encoded = encoded * effective.unsqueeze(-1).to(encoded.dtype)
        return encoded, masked_mean(encoded, effective.to(encoded.dtype))


class Heads(nn.Module):
    """分类 + 回归双任务头（回归不加 tanh，避免强度向 0 收缩）。"""

    def __init__(self, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                                   nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden, len(CLASS_NAMES))
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, fused: torch.Tensor):
        fused = self.trunk(fused)
        return self.classifier(fused), self.regressor(fused).squeeze(-1), fused


def modality_stats(batch: dict) -> torch.Tensor:
    """每个模态的 [观测比例, 有效轴比例, 缺失比例]，形状 (B, 3, 3)。"""
    stats = []
    for name in MODALITIES:
        observed, exist = batch[f"{name}_observed"], batch[f"{name}_exist"]
        stats.append(torch.stack([observed.mean(dim=1), exist.mean(dim=1),
                                  (exist * (1 - observed)).mean(dim=1)], dim=1))
    return torch.stack(stats, dim=1)


def observation_weights(stats: torch.Tensor) -> torch.Tensor:
    """无门控模型用的朴素权重：按观测比例归一化。"""
    weights = stats[:, :, 0].clamp_min(1e-6)
    return weights / weights.sum(dim=1, keepdim=True)


class EarlyFusionModel(nn.Module):
    """Baseline 1：早期融合 + 掩码均值池化（性能下限参考）。"""

    name = "early"

    def __init__(self, hidden: int = 96, dropout: float = 0.2, **_):
        super().__init__()
        self.projection = nn.ModuleDict({m: nn.Linear(INPUT_DIMS[m], hidden) for m in MODALITIES})
        self.heads = Heads(hidden * 3, dropout)

    def encode(self, batch: dict) -> dict:
        pooled = {}
        for name in MODALITIES:
            hidden = F.relu(self.projection[name](batch[name]))
            pooled[name] = masked_mean(hidden, batch[f"{name}_observed"])
        return pooled

    def forward(self, batch: dict, teacher_features: dict | None = None, **_) -> dict:
        pooled = self.encode(batch)
        fused = torch.cat([pooled[name] for name in MODALITIES], dim=-1)
        logits, intensity, _ = self.heads(fused)
        stats = modality_stats(batch)
        return {"logits": logits, "intensity": intensity, "weights": observation_weights(stats),
                "ratios": stats[:, :, 0], "pooled": pooled, "stats": stats}


class CrossAttentionFusionModel(nn.Module):
    """Baseline 2：跨模态注意力（简化 MulT），同时作为知识蒸馏的 Teacher。"""

    name = "cross_attn"

    def __init__(self, hidden: int = 96, layers: int = 2, nhead: int = 4, dropout: float = 0.2, **_):
        super().__init__()
        self.encoders = nn.ModuleDict({m: ModalEncoder(INPUT_DIMS[m], hidden, layers, nhead, dropout)
                                       for m in MODALITIES})
        self.cross = nn.ModuleDict({m: nn.MultiheadAttention(hidden, nhead, batch_first=True,
                                                             dropout=dropout) for m in MODALITIES})
        self.norms = nn.ModuleDict({m: nn.LayerNorm(hidden) for m in MODALITIES})
        self.heads = Heads(hidden * 3, dropout)

    def encode(self, batch: dict):
        tokens, pooled = {}, {}
        for name in MODALITIES:
            tokens[name], pooled[name] = self.encoders[name](
                batch[name], batch[f"{name}_exist"], batch[f"{name}_observed"], batch[f"{name}_missing"])
        return tokens, pooled

    def forward(self, batch: dict, teacher_features: dict | None = None, **_) -> dict:
        tokens, pooled = self.encode(batch)
        refined = {}
        for name in MODALITIES:
            others = [other for other in MODALITIES if other != name]
            key_value = torch.cat([tokens[other] for other in others], dim=1)
            key_padding = torch.cat([batch[f"{other}_observed"] <= 0 for other in others], dim=1)
            # 若某个样本的另外两个模态都完全没有观测（整模态缺失），解除屏蔽以保证注意力数值稳定
            empty_row = ~key_padding.any(dim=1, keepdim=True)
            key_padding = torch.where(empty_row, torch.zeros_like(key_padding), key_padding)
            updated, _ = self.cross[name](tokens[name], key_value, key_value,
                                          key_padding_mask=key_padding, need_weights=False)
            refined[name] = self.norms[name](tokens[name] + updated)
        fused = torch.cat([masked_mean(refined[name], batch[f"{name}_observed"]) for name in MODALITIES],
                          dim=-1)
        logits, intensity, _ = self.heads(fused)
        stats = modality_stats(batch)
        return {"logits": logits, "intensity": intensity, "weights": observation_weights(stats),
                "ratios": stats[:, :, 0], "pooled": pooled, "tokens": refined, "stats": stats}


class MisaLikeModel(nn.Module):
    """Baseline 3：不变表示 + 特异表示分解；缺失模态的特异表示置零，不变表示保留。"""

    name = "misa"

    def __init__(self, hidden: int = 96, layers: int = 1, nhead: int = 4, dropout: float = 0.2, **_):
        super().__init__()
        self.shared = nn.ModuleDict({m: nn.Linear(INPUT_DIMS[m], hidden) for m in MODALITIES})
        self.private = nn.ModuleDict({m: nn.Linear(INPUT_DIMS[m], hidden) for m in MODALITIES})
        self.encoders = nn.ModuleDict({m: ModalEncoder(hidden, hidden, layers, nhead, dropout)
                                       for m in MODALITIES})
        self.heads = Heads(hidden * 6, dropout)

    def forward(self, batch: dict, teacher_features: dict | None = None, **_) -> dict:
        invariant, specific = {}, {}
        for name in MODALITIES:
            observed = batch[f"{name}_observed"]
            shared_tokens, _ = self.encoders[name](self.shared[name](batch[name]), batch[f"{name}_exist"],
                                                   observed, batch[f"{name}_missing"])
            private = self.private[name](batch[name])
            has_observation = (observed.sum(dim=1, keepdim=True) > 0).to(private.dtype)
            invariant[name] = masked_mean(shared_tokens, observed)
            specific[name] = masked_mean(private * observed.unsqueeze(-1), observed) * has_observation
        fused = torch.cat([torch.cat([invariant[name], specific[name]], dim=-1) for name in MODALITIES],
                          dim=-1)
        logits, intensity, _ = self.heads(fused)
        stats = modality_stats(batch)
        return {"logits": logits, "intensity": intensity, "weights": observation_weights(stats),
                "ratios": stats[:, :, 0], "pooled": invariant, "invariant": invariant,
                "specific": specific, "stats": stats}


class LocalContinuousReconstruction(nn.Module):
    """创新点 2：局部连续重建（LCR）。

    对每个缺失位置，只在“前后各 window 个时间位置”内、且**有观测**的 token 上做跨模态注意力：
    - 目标模态的邻域上下文（前后窗口）
    - 另外两个模态在对应时段的观测 token
    查询向量由“时间位置嵌入 + 模态嵌入”构成（缺失位置原始值为零、不可靠，故不作为 query 值）。
    window 设为序列长度时退化为“全局重建”，可用于消融对比。
    """

    def __init__(self, hidden: int, nhead: int = 4, dropout: float = 0.2, max_steps: int = 512):
        super().__init__()
        self.query_pos = nn.Embedding(max_steps, hidden)
        self.query_mod = nn.Embedding(len(MODALITIES), hidden)
        self.attention = nn.MultiheadAttention(hidden, nhead, batch_first=True, dropout=dropout)
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, tokens: dict, observed: dict, missing: dict, window: int | None):
        batch, steps, hidden = tokens[MODALITIES[0]].shape
        window = steps if window is None else max(1, int(window))
        index = torch.arange(steps, device=tokens[MODALITIES[0]].device)
        keys_all = torch.cat([tokens[name] for name in MODALITIES], dim=1)          # (B, 3T, H)
        key_valid = torch.cat([observed[name] > 0 for name in MODALITIES], dim=1)   # (B, 3T)
        distance = (index[:, None] - index[None, :]).abs() <= window                # (T, T)
        reconstructions, confidences, valid_masks = {}, {}, {}
        for position, name in enumerate(MODALITIES):
            query = (self.query_pos(index) + self.query_mod.weight[position]).unsqueeze(0).expand(batch, -1, -1)
            allowed = []
            for other_index, other in enumerate(MODALITIES):
                block = key_valid[:, other_index * steps:(other_index + 1) * steps]
                local = distance
                if other == name:
                    local = local & (index[:, None] != index[None, :])   # 不利用缺失位置自身的零值
                allowed.append(block.unsqueeze(1) & local.unsqueeze(0))
            allowed = torch.cat(allowed, dim=2)                                   # (B, T, 3T)
            fallback = torch.zeros_like(allowed)
            fallback[:, :, 0] = True
            allowed = torch.where(allowed.any(dim=2, keepdim=True), allowed, fallback)
            context = allowed.sum(dim=2).to(tokens[name].dtype)
            mask = torch.where(allowed, torch.zeros_like(allowed, dtype=tokens[name].dtype),
                               torch.full_like(allowed, float("-inf"), dtype=tokens[name].dtype))
            mask = mask.unsqueeze(1).expand(batch, self.attention.num_heads, steps, -1).reshape(
                batch * self.attention.num_heads, steps, -1)
            attended, _ = self.attention(query, keys_all, keys_all, attn_mask=mask, need_weights=False)
            reconstructions[name] = self.output(attended)
            confidences[name] = (context / float(2 * window + 2)).clamp(max=1.0)
            valid_masks[name] = (missing[name] > 0) & (context > 0)
        return reconstructions, confidences, valid_masks


def reduce_confidence(value, reference: torch.Tensor) -> torch.Tensor:
    """把重建置信度归约成 (B,)；LCR 关闭时 value 为 None，返回全 0。"""
    if value is None:
        return torch.zeros_like(reference)
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if value.dim() > 1:
        value = value.mean(dim=1)
    return value.reshape(reference.shape[0])


class QualityAwareGate(nn.Module):
    """创新点 3：质量门控融合。g_m = sigmoid(W[h_m; r_m])，r_m 含观测/缺失/重建置信度。"""

    def __init__(self, hidden: int, dropout: float = 0.2, width: int = 48):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(hidden + 4, width), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, pooled: dict, stats: torch.Tensor, confidence: dict):
        hidden = torch.stack([pooled[name] for name in MODALITIES], dim=1)       # (B, 3, H)
        # 重建置信度统一归约为 (B,)；无 LCR 时 confidence 为空字典，此时该项恒为 0
        reference = stats[:, 0, 0]
        confidence_stack = torch.stack([reduce_confidence(confidence.get(name), reference)
                                        for name in MODALITIES], dim=1)          # (B, 3)
        # 每模态单独打分：[h_m, 观测比例, 有效轴比例, 缺失比例, 重建置信度]
        features = torch.cat([hidden,
                              stats[:, :, 0].unsqueeze(-1),                  # 观测比例
                              stats[:, :, 1].unsqueeze(-1),                  # 有效轴比例（长度边界）
                              stats[:, :, 2].unsqueeze(-1),                  # 缺失比例
                              confidence_stack.unsqueeze(-1)], dim=-1)       # (B, 3, H+4)
        scores = self.network(features).squeeze(-1)                          # (B, 3)
        available = stats[:, :, 0] > 0
        scores = scores.masked_fill(~available, -1e4)
        weights = torch.softmax(scores, dim=1)
        return weights * available.to(weights.dtype), hidden


class DualMaskCrossModalReconstructionNet(nn.Module):
    """主模型 DM-CMRN：双掩码编码 + 局部连续重建 + 质量门控融合。

    消融开关：
        use_dual_mask=False  退化为“只用 padding mask”（缺失位置被当作填充）
        use_lcr=False        去掉局部连续重建
        rec_mode="global"    全局重建（窗口不受限）而非局部重建
        use_qagf=False       用“观测比例”朴素权重替代质量门控
    """

    name = "dm_cmrn"

    def __init__(self, hidden: int = 96, layers: int = 2, nhead: int = 4, dropout: float = 0.2,
                 use_dual_mask: bool = True, use_lcr: bool = True, rec_mode: str = "local",
                 ctx_span: int = 6, use_qagf: bool = True, max_steps: int = 512, **_):
        super().__init__()
        self.use_dual_mask = use_dual_mask
        self.use_lcr = use_lcr
        self.rec_mode = rec_mode
        self.ctx_span = ctx_span
        self.use_qagf = use_qagf
        self.encoders = nn.ModuleDict({
            m: ModalEncoder(INPUT_DIMS[m], hidden, layers, nhead, dropout,
                            use_status_embed=use_dual_mask) for m in MODALITIES})
        self.reconstruction = (LocalContinuousReconstruction(hidden, nhead, dropout, max_steps)
                               if use_lcr else None)
        self.gate = QualityAwareGate(hidden, dropout) if use_qagf else None
        self.fusion = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                                    nn.GELU(), nn.Dropout(dropout))
        self.heads = Heads(hidden, dropout)

    def forward(self, batch: dict, teacher_features: dict | None = None, **_) -> dict:
        observed_used = {name: (batch[f"{name}_observed"] if self.use_dual_mask
                                else batch[f"{name}_exist"]) for name in MODALITIES}
        missing_used = {name: (batch[f"{name}_missing"] if self.use_dual_mask
                               else torch.zeros_like(batch[f"{name}_exist"])) for name in MODALITIES}
        tokens, pooled = {}, {}
        for name in MODALITIES:
            tokens[name], pooled[name] = self.encoders[name](
                batch[name], batch[f"{name}_exist"], observed_used[name], missing_used[name])
        stats = modality_stats(batch)

        reconstruction, confidence, valid = None, {}, {}
        encoded_tokens = dict(tokens)                    # 补全前的编码表示（供 context 重建目标使用）
        if self.reconstruction is not None:
            window = None if self.rec_mode == "global" else self.ctx_span
            reconstruction, confidence, valid = self.reconstruction(
                tokens, observed_used, missing_used, window)
            for name in MODALITIES:                     # 只用有效重建结果补全缺失位置
                filled = torch.where(valid[name].unsqueeze(-1), reconstruction[name], tokens[name])
                coverage = torch.clamp(observed_used[name] + valid[name].to(observed_used[name].dtype), 0, 1)
                pooled[name] = masked_mean(filled, coverage)
                tokens[name] = filled

        if self.gate is not None:
            weights, hidden_stack = self.gate(pooled, stats, confidence)
        else:
            weights = observation_weights(stats)
            hidden_stack = torch.stack([pooled[name] for name in MODALITIES], dim=1)
        fused = self.fusion((hidden_stack * weights.unsqueeze(-1)).sum(dim=1))
        logits, intensity, _ = self.heads(fused)
        return {"logits": logits, "intensity": intensity, "weights": weights, "ratios": stats[:, :, 0],
                "pooled": pooled, "tokens": tokens, "tokens_encoded": encoded_tokens, "stats": stats,
                "rec": reconstruction, "rec_valid": valid, "rec_conf": confidence}

    @torch.no_grad()
    def encode_clean(self, batch: dict) -> dict:
        """用“干净输入 + 完整有效轴”编码，作为自重建目标（与重建输出同一隐空间）。"""
        tokens = {}
        for name in MODALITIES:
            observed = batch[f"{name}_exist"]
            zeros = torch.zeros_like(observed)
            tokens[name], _ = self.encoders[name](batch[f"{name}_clean"], observed, observed, zeros)
        return tokens


MODEL_REGISTRY = {
    "early": EarlyFusionModel,
    "cross_attn": CrossAttentionFusionModel,
    "misa": MisaLikeModel,
    "dm_cmrn": DualMaskCrossModalReconstructionNet,
}

# 消融方案：键为实验名，值为 build_model 的额外参数
ABLATION_VARIANTS = {
    "dm_cmrn_full": {},
    "dm_cmrn_no_dual_mask": {"use_dual_mask": False},
    "dm_cmrn_no_lcr": {"use_lcr": False},
    "dm_cmrn_global_rec": {"rec_mode": "global"},
    "dm_cmrn_no_qagf": {"use_qagf": False},
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型：{name}，可选 {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)




