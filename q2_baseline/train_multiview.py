import argparse
import copy
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MODALITIES = ("text", "audio", "vision")
INPUT_DIMS = {"text": 768, "audio": 74, "vision": 35}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def mask_for(split, name, kind):
    values = np.asarray(split[name])
    if name == "text":
        return np.asarray(split["text_bert"])[:, 1, :] != 0
    if kind == "unaligned" and name + "_lengths" in split:
        lengths = np.asarray(split[name + "_lengths"], dtype=np.int64)
        return np.arange(values.shape[1])[None, :] < lengths[:, None]
    if name + "_mask" in split:
        return np.asarray(split[name + "_mask"], dtype=bool)
    return np.any(np.isfinite(values) & (values != 0), axis=-1)


def compute_normalizers(split, kind):
    normalizers = {}
    for name in MODALITIES:
        values = np.asarray(split[name], dtype=np.float32)
        mask = mask_for(split, name, kind)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        present = values[mask]
        mean = present.mean(axis=0).astype(np.float32)
        std = np.maximum(present.std(axis=0), 1e-4).astype(np.float32)
        normalizers[name] = (mean, std)
    return normalizers


def convert_split(split, kind, normalizers):
    features, masks = {}, {}
    for name in MODALITIES:
        values = np.asarray(split[name], dtype=np.float32)
        mask = mask_for(split, name, kind)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mean, std = normalizers[name]
        values = np.clip((values - mean) / std, -10.0, 10.0)
        values *= mask[..., None]
        features[name] = torch.from_numpy(values)
        masks[name] = torch.from_numpy(mask)
    targets = {
        "class": torch.from_numpy(np.asarray(split["classification_labels"], dtype=np.int64)),
        "regression": torch.from_numpy(np.asarray(split["regression_labels"], dtype=np.float32)),
    }
    return features, masks, targets, list(split["id"])


def load_view(path, kind):
    with Path(path).open("rb") as handle:
        source = pickle.load(handle)
    normalizers = compute_normalizers(source["train"], kind)
    splits = {
        name: convert_split(split, kind, normalizers)
        for name, split in source.items()
    }
    del source
    return {"kind": kind, "normalizers": normalizers, "splits": splits}


def validate_views(aligned, unaligned):
    for split in ("train", "valid", "test"):
        aligned_ids = aligned["splits"][split][3]
        unaligned_ids = unaligned["splits"][split][3]
        if aligned_ids != unaligned_ids:
            raise ValueError(f"ID order differs for {split}")


def batches(dataset, batch_size, device, shuffle, seed):
    features, masks, targets, _ = dataset
    count = len(targets["class"])
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(count, generator=generator) if shuffle else torch.arange(count)
    for start in range(0, count, batch_size):
        positions = indices[start:start + batch_size]
        yield (
            {name: features[name][positions].to(device, non_blocking=True) for name in MODALITIES},
            {name: masks[name][positions].to(device, non_blocking=True) for name in MODALITIES},
            {name: value[positions].to(device, non_blocking=True) for name, value in targets.items()},
        )


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout, bidirectional=False):
        super().__init__()
        if bidirectional and hidden % 2:
            raise ValueError("--hidden must be even when --bidirectional is enabled")
        self.projection = nn.Linear(input_dim, hidden)
        gru_hidden = hidden // 2 if bidirectional else hidden
        self.gru = nn.GRU(hidden, gru_hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0,
                          bidirectional=bidirectional)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, mask):
        encoded = F.gelu(self.projection(values))
        encoded = encoded * mask.unsqueeze(-1)
        temporal, _ = self.gru(encoded)
        tokens = self.norm(encoded + self.dropout(temporal))
        tokens = tokens * mask.unsqueeze(-1)
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return tokens, pooled


class ModalitySSLModel(nn.Module):
    def __init__(self, name, hidden, layers, dropout, projector_hidden, bidirectional=False):
        super().__init__()
        self.name = name
        self.encoder = TemporalEncoder(INPUT_DIMS[name], hidden, layers, dropout, bidirectional)
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, projector_hidden),
            nn.GELU(),
            nn.Linear(projector_hidden, projector_hidden),
        )

    def forward(self, values, mask):
        _, representation = self.encoder(values, mask)
        return representation, self.projector(representation)


class FusionModel(nn.Module):
    def __init__(self, hidden, layers, dropout, bidirectional=False):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: TemporalEncoder(INPUT_DIMS[name], hidden, layers, dropout, bidirectional)
            for name in MODALITIES
        })
        self.residual_projections = nn.ModuleDict({
            name: nn.Linear(INPUT_DIMS[name] * 2, hidden)
            for name in MODALITIES
        })
        for projection in self.residual_projections.values():
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        fusion_hidden = hidden * 2
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 6, fusion_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
        self.regressor = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def encode(self, features, masks):
        representations = [self.encoders[name](features[name], masks[name])[1] for name in MODALITIES]
        residuals = []
        for name in MODALITIES:
            values = features[name]
            weights = masks[name].unsqueeze(-1).to(values.dtype)
            count = weights.sum(dim=1).clamp_min(1.0)
            mean = (values * weights).sum(dim=1) / count
            centered = (values - mean.unsqueeze(1)) * weights
            std = torch.sqrt((centered.square()).sum(dim=1) / count + 1e-4)
            residuals.append(self.residual_projections[name](torch.cat((mean, std), dim=-1)))
        return torch.cat((*representations, *residuals), dim=-1)

    def forward(self, features, masks):
        fused = self.fusion(self.encode(features, masks))
        return self.classifier(fused), self.regressor(fused).squeeze(-1), fused


def metrics(truth, predicted, actual, estimate):
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    estimate = np.asarray(estimate)
    f1_values = []
    for label in range(3):
        true_positive = np.sum((truth == label) & (predicted == label))
        false_positive = np.sum((truth != label) & (predicted == label))
        false_negative = np.sum((truth == label) & (predicted != label))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(float(2 * true_positive / denominator) if denominator else 0.0)
    pearson = float(np.corrcoef(actual, estimate)[0, 1]) if np.std(actual) > 0 and np.std(estimate) > 0 else 0.0
    return {
        "accuracy": float(np.mean(truth == predicted)),
        "f1_macro": float(np.mean(f1_values)),
        "mae": float(np.mean(np.abs(actual - estimate))),
        "pearson": pearson,
    }


def supervised_loss(logits, regression, targets, regression_weight):
    return F.cross_entropy(logits, targets["class"]) + regression_weight * F.smooth_l1_loss(regression, targets["regression"])


def amp_context(enabled):
    return torch.cuda.amp.autocast(enabled=enabled)


def augment_temporal_view(values, mask, feature_noise, time_mask_probability):
    augmented_mask = mask.clone()
    keep = torch.rand(mask.shape, device=values.device) >= time_mask_probability
    augmented_mask &= keep
    empty = ~augmented_mask.any(dim=1)
    if empty.any():
        valid_positions = mask[empty].float().argmax(dim=1)
        rows = torch.nonzero(empty, as_tuple=False).squeeze(1)
        augmented_mask[rows, valid_positions] = True
    noise = torch.randn_like(values) * feature_noise
    augmented = (values + noise) * augmented_mask.unsqueeze(-1).to(values.dtype)
    return augmented, augmented_mask


def off_diagonal(matrix):
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def vicreg_loss(first, second, variance_weight, covariance_weight, variance_floor):
    invariance = F.mse_loss(first, second)
    first_centered = first - first.mean(dim=0)
    second_centered = second - second.mean(dim=0)
    first_std = torch.sqrt(first_centered.var(dim=0, unbiased=False) + 1e-04)
    second_std = torch.sqrt(second_centered.var(dim=0, unbiased=False) + 1e-04)
    variance = 0.5 * (
        F.relu(variance_floor - first_std).mean()
        + F.relu(variance_floor - second_std).mean()
    )
    first_covariance = first_centered.T @ first_centered / max(first.shape[0] - 1, 1)
    second_covariance = second_centered.T @ second_centered / max(second.shape[0] - 1, 1)
    covariance = 0.5 * (
        off_diagonal(first_covariance).pow(2).mean()
        + off_diagonal(second_covariance).pow(2).mean()
    )
    return invariance + variance_weight * variance + covariance_weight * covariance


def ssl_batch_loss(model, features, masks, feature_noise, time_mask_probability,
                   variance_weight, covariance_weight, variance_floor):
    values = features[model.name]
    mask = masks[model.name]
    first_values, first_mask = augment_temporal_view(values, mask, feature_noise, time_mask_probability)
    second_values, second_mask = augment_temporal_view(values, mask, feature_noise, time_mask_probability)
    _, first_projection = model(first_values, first_mask)
    _, second_projection = model(second_values, second_mask)
    return vicreg_loss(first_projection, second_projection, variance_weight, covariance_weight, variance_floor)


@torch.no_grad()
def evaluate_ssl(model, dataset, batch_size, device, amp_enabled, args):
    model.eval()
    total = 0.0
    steps = 0
    for features, masks, _ in batches(dataset, batch_size, device, False, 0):
        with amp_context(amp_enabled):
            loss = ssl_batch_loss(model, features, masks, args.feature_noise,
                                  args.time_mask_probability, args.variance_weight,
                                  args.covariance_weight, args.variance_floor)
        total += loss.item()
        steps += 1
    return total / max(steps, 1)


def mask_interval(features, masks, modality, start, length):
    updated_masks = {name: value.clone() for name, value in masks.items()}
    updated_masks[modality][:, start:start + length] = False
    updated_features = {name: value.clone() for name, value in features.items()}
    updated_features[modality] *= updated_masks[modality].unsqueeze(-1)
    return updated_features, updated_masks


@torch.no_grad()
def evaluate_fusion(model, dataset, batch_size, device, amp_enabled, scenario=None):
    model.eval()
    classes, strengths = [], []
    targets = dataset[2]
    for features, masks, _ in batches(dataset, batch_size, device, False, 0):
        if scenario is not None:
            features, masks = mask_interval(features, masks, *scenario)
        with amp_context(amp_enabled):
            logits, regression, _ = model(features, masks)
        classes.extend(logits.argmax(dim=-1).cpu().tolist())
        strengths.extend(regression.cpu().tolist())
    return metrics(targets["class"].numpy(), classes, targets["regression"].numpy(), strengths)


def scenario_matrix():
    scenarios = {"clean": None}
    for name in MODALITIES:
        for location, start_for_length in (("start", lambda length: 0),
                                           ("middle", lambda length: (50 - length) // 2),
                                           ("end", lambda length: 50 - length)):
            for length in (5, 15, 25):
                scenarios[f"{name}_{location}_{length}"] = (name, start_for_length(length), length)
    return scenarios


def fusion_selection(model, dataset, batch_size, device, amp_enabled):
    clean = evaluate_fusion(model, dataset, batch_size, device, amp_enabled)
    missing = [evaluate_fusion(model, dataset, batch_size, device, amp_enabled, (name, 17, 15))
               for name in MODALITIES]
    mean_f1 = np.mean([clean["f1_macro"], *(score["f1_macro"] for score in missing)])
    mean_mae = np.mean([clean["mae"], *(score["mae"] for score in missing)])
    return float(mean_f1 - 0.1 * mean_mae)


def load_encoder_states(model, directory):
    directory = Path(directory)
    for name in MODALITIES:
        checkpoint_path = directory / "encoders" / f"encoder_{name}.pt"
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.encoders[name].load_state_dict(checkpoint["encoder"])


def train_single_modality(name, view, output, args, device, scaler):
    train_data = view["splits"]["train"]
    valid_data = view["splits"]["valid"]
    model = ModalitySSLModel(name, args.hidden, args.layers, args.dropout,
                             args.projector_hidden, args.bidirectional).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.encoder_learning_rate, weight_decay=1e-4)
    best_score = float("inf")
    best_state = None
    for epoch in range(1, args.encoder_epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for features, masks, _ in batches(train_data, args.batch_size, device, True, args.seed + epoch):
            with amp_context(args.amp):
                loss = ssl_batch_loss(model, features, masks, args.feature_noise,
                                      args.time_mask_probability, args.variance_weight,
                                      args.covariance_weight, args.variance_floor)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            steps += 1
        validation_loss = evaluate_ssl(model, valid_data, args.batch_size, device, args.amp, args)
        print(f"modality={name} ssl_epoch={epoch} train_ssl={total / max(steps, 1):.4f} valid_ssl={validation_loss:.4f} labels=unused", flush=True)
        if validation_loss < best_score:
            best_score = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.encoder.state_dict().items()}
    encoder_dir = output / "encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": best_state, "modality": name, "validation_ssl_loss": best_score,
                "config": vars(args), "label_free": True}, encoder_dir / f"encoder_{name}.pt")


def train_separate_fusion(aligned, output, args, device, scaler, encoder_source=None):
    model = FusionModel(args.hidden, args.layers, args.dropout, args.bidirectional).to(device)
    load_encoder_states(model, encoder_source or output)
    for parameter in model.encoders.parameters():
        parameter.requires_grad_(False)
    task_parameters = list(model.residual_projections.parameters())
    task_parameters.extend(model.fusion.parameters())
    task_parameters.extend(model.classifier.parameters())
    task_parameters.extend(model.regressor.parameters())
    optimizer = torch.optim.Adam([
        {"params": model.encoders.parameters(), "lr": args.fusion_encoder_learning_rate},
        {"params": task_parameters, "lr": args.fusion_learning_rate},
    ], weight_decay=1e-4)
    train_data = aligned["splits"]["train"]
    valid_data = aligned["splits"]["valid"]
    best_score = float("-inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.fusion_epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for parameter in model.encoders.parameters():
                parameter.requires_grad_(True)
        model.train()
        total = 0.0
        steps = 0
        for features, masks, targets in batches(train_data, args.batch_size, device, True, args.seed + 1000 + epoch):
            with amp_context(args.amp):
                logits, regression, _ = model(features, masks)
                loss = supervised_loss(logits, regression, targets, args.regression_weight)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            steps += 1
        score = fusion_selection(model, valid_data, args.batch_size, device, args.amp)
        print(f"fusion epoch={epoch} loss={total / steps:.4f} selection={score:.6f}", flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    results = {name: evaluate_fusion(model, valid_data, args.batch_size, device, args.amp, scenario)
               for name, scenario in scenario_matrix().items()}
    torch.save({"model": best_state, "normalizers": aligned["normalizers"],
                "config": vars(args), "best_epoch": best_epoch, "results": results}, output / "fusion_best.pt")
    (output / "fusion_validation.json").write_text(json.dumps({"best_epoch": best_epoch, "selection": best_score, "results": results}, indent=2), encoding="utf-8")


def train_hybrid(aligned, unaligned, output, args, device, scaler):
    model = FusionModel(args.hidden, args.layers, args.dropout, args.bidirectional).to(device)
    if args.init_dir:
        load_encoder_states(model, args.init_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.hybrid_learning_rate, weight_decay=1e-4)
    aligned_train = aligned["splits"]["train"]
    unaligned_train = unaligned["splits"]["train"]
    aligned_valid = aligned["splits"]["valid"]
    best_score = float("-inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.hybrid_epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        aligned_batches = batches(aligned_train, args.hybrid_batch_size, device, True, args.seed + 2000 + epoch)
        unaligned_batches = batches(unaligned_train, args.hybrid_batch_size, device, True, args.seed + 2000 + epoch)
        for aligned_batch, unaligned_batch in zip(aligned_batches, unaligned_batches):
            aligned_features, aligned_masks, targets = aligned_batch
            unaligned_features, unaligned_masks, unaligned_targets = unaligned_batch
            if not torch.equal(targets["class"].cpu(), unaligned_targets["class"].cpu()):
                raise RuntimeError("hybrid batches lost aligned sample order")
            with amp_context(args.amp):
                aligned_logits, aligned_regression, aligned_rep = model(aligned_features, aligned_masks)
                unaligned_logits, unaligned_regression, unaligned_rep = model(unaligned_features, unaligned_masks)
                supervised = 0.5 * (
                    supervised_loss(aligned_logits, aligned_regression, targets, args.regression_weight)
                    + supervised_loss(unaligned_logits, unaligned_regression, targets, args.regression_weight)
                )
                representation_loss = 1.0 - F.cosine_similarity(aligned_rep, unaligned_rep, dim=-1).mean()
                logit_loss = F.mse_loss(aligned_logits, unaligned_logits)
                regression_loss = F.smooth_l1_loss(aligned_regression, unaligned_regression)
                loss = supervised + args.consistency_weight * (
                    representation_loss + args.logit_consistency_weight * logit_loss + regression_loss
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            steps += 1
        score = fusion_selection(model, aligned_valid, args.hybrid_batch_size, device, args.amp)
        print(f"hybrid epoch={epoch} loss={total / steps:.4f} selection={score:.6f}", flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    results = {name: evaluate_fusion(model, aligned_valid, args.hybrid_batch_size, device, args.amp, scenario)
               for name, scenario in scenario_matrix().items()}
    torch.save({"model": best_state, "aligned_normalizers": aligned["normalizers"],
                "unaligned_normalizers": unaligned["normalizers"], "config": vars(args),
                "best_epoch": best_epoch, "results": results}, output / "hybrid_best.pt")
    (output / "hybrid_validation.json").write_text(json.dumps({"best_epoch": best_epoch, "selection": best_score, "results": results}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Separate unaligned encoders plus aligned fusion, or two-view hybrid training")
    parser.add_argument("--mode", choices=("separate", "fusion", "hybrid"), required=True)
    parser.add_argument("--aligned-data", type=Path, required=True)
    parser.add_argument("--unaligned-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-dir", type=Path)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--bidirectional", action="store_true",
                        help="use a bidirectional GRU while preserving the encoder output width")
    parser.add_argument("--projector-hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hybrid-batch-size", type=int, default=32)
    parser.add_argument("--encoder-epochs", type=int, default=10)
    parser.add_argument("--fusion-epochs", type=int, default=20)
    parser.add_argument("--hybrid-epochs", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--encoder-learning-rate", type=float, default=3e-4)
    parser.add_argument("--fusion-learning-rate", type=float, default=2e-4)
    parser.add_argument("--fusion-encoder-learning-rate", type=float, default=5e-5)
    parser.add_argument("--hybrid-learning-rate", type=float, default=1e-4)
    parser.add_argument("--regression-weight", type=float, default=0.5)
    parser.add_argument("--feature-noise", type=float, default=0.05)
    parser.add_argument("--time-mask-probability", type=float, default=0.15)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--covariance-weight", type=float, default=0.04)
    parser.add_argument("--variance-floor", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--logit-consistency-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.mode in ("fusion", "hybrid") and args.init_dir is None:
        parser.error(f"{args.mode} mode requires --init-dir from the separate experiment")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp_enabled = args.amp and device.type == "cuda"
    print(f"device={device} amp={amp_enabled} mode={args.mode}", flush=True)
    aligned = load_view(args.aligned_data, "aligned")
    unaligned = load_view(args.unaligned_data, "unaligned")
    validate_views(aligned, unaligned)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if args.mode == "separate":
        for name in MODALITIES:
            train_single_modality(name, unaligned, args.output, args, device, scaler)
        train_separate_fusion(aligned, args.output, args, device, scaler)
    elif args.mode == "fusion":
        train_separate_fusion(aligned, args.output, args, device, scaler, args.init_dir)
    else:
        train_hybrid(aligned, unaligned, args.output, args, device, scaler)
    (args.output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print(f"done output={args.output}", flush=True)


if __name__ == "__main__":
    main()
