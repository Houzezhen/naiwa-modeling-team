import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MODALITIES = ("text", "audio", "vision")


def prepare_split(split, normalizers=None):
    features = {}
    masks = {}
    for name in MODALITIES:
        values = np.asarray(split[name], dtype=np.float32)
        if values.ndim != 3 or values.shape[1] != 50:
            raise ValueError(f"Unexpected {name} shape: {values.shape}")
        if name == "text":
            bert = np.asarray(split["text_bert"])
            mask = bert[:, 1, :] != 0
        elif name + "_mask" in split:
            mask = np.asarray(split[name + "_mask"], dtype=bool)
            if mask.shape != values.shape[:2]:
                raise ValueError(f"Unexpected {name}_mask shape: {mask.shape}")
        else:
            mask = np.any(np.isfinite(values) & (values != 0), axis=-1)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if normalizers is None:
            present = values[mask]
            mean = present.mean(axis=0).astype(np.float32)
            std = np.maximum(present.std(axis=0), 1e-4).astype(np.float32)
            stats = (mean, std)
        else:
            stats = normalizers[name]
        values = np.clip((values - stats[0]) / stats[1], -10, 10)
        values *= mask[..., None]
        features[name] = torch.from_numpy(values)
        masks[name] = torch.from_numpy(mask)
        if normalizers is None:
            masks[name + "_stats"] = stats
    labels = np.asarray(split["classification_labels"])
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Classification labels must be 0, 1, 2")
    targets = {
        "class": torch.from_numpy(labels.astype(np.int64)),
        "regression": torch.from_numpy(np.asarray(split["regression_labels"], dtype=np.float32)),
    }
    return features, masks, targets


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, mask):
        encoded = F.relu(self.projection(values))
        encoded = encoded * mask.unsqueeze(-1)
        temporal = self.conv(encoded.transpose(1, 2)).transpose(1, 2)
        encoded = self.norm(encoded + self.dropout(F.relu(temporal)))
        encoded = encoded * mask.unsqueeze(-1)
        return encoded.sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)


class Baseline(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: Encoder(size, hidden, dropout)
            for name, size in (("text", 768), ("audio", 74), ("vision", 35))
        })
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden, 3)
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, features, masks):
        fused = self.fusion(torch.cat([
            self.encoders[name](features[name], masks[name]) for name in MODALITIES
        ], dim=-1))
        return self.classifier(fused), self.regressor(fused).squeeze(-1)


def batches(features, masks, targets, batch_size, device, shuffle):
    indices = torch.randperm(len(targets["class"])) if shuffle else torch.arange(len(targets["class"]))
    for start in range(0, len(indices), batch_size):
        positions = indices[start:start + batch_size]
        yield (
            {name: features[name][positions].to(device) for name in MODALITIES},
            {name: masks[name][positions].to(device) for name in MODALITIES},
            {name: value[positions].to(device) for name, value in targets.items()},
        )


def metrics(truth, predicted, actual, estimate):
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    estimate = np.asarray(estimate)
    per_class = []
    for label in range(3):
        true_positive = np.sum((truth == label) & (predicted == label))
        false_positive = np.sum((truth != label) & (predicted == label))
        false_negative = np.sum((truth == label) & (predicted != label))
        denominator = 2 * true_positive + false_positive + false_negative
        per_class.append(float(2 * true_positive / denominator) if denominator else 0.0)
    pearson = float(np.corrcoef(actual, estimate)[0, 1]) if np.std(actual) > 0 and np.std(estimate) > 0 else 0.0
    return {
        "accuracy": float(np.mean(truth == predicted)),
        "f1_macro": float(np.mean(per_class)),
        "mae": float(np.mean(np.abs(actual - estimate))),
        "pearson": pearson,
    }


@torch.no_grad()
def evaluate(model, dataset, batch_size, device):
    model.eval()
    features, masks, targets = dataset
    classes, strengths = [], []
    for batch_features, batch_masks, _ in batches(features, masks, targets, batch_size, device, False):
        logits, regression = model(batch_features, batch_masks)
        classes.extend(logits.argmax(dim=-1).cpu().tolist())
        strengths.extend(regression.cpu().tolist())
    scores = metrics(targets["class"].numpy(), classes, targets["regression"].numpy(), strengths)
    return scores, classes, strengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--regression-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.device == "auto":
        device = torch.device("cpu")
        for index in range(torch.cuda.device_count()):
            candidate = torch.device(f"cuda:{index}")
            try:
                torch.relu(torch.ones(1, device=candidate)).item()
            except RuntimeError:
                continue
            device = candidate
            break
    else:
        device = torch.device(args.device)
    print(f"device={device}; loading {args.data}", flush=True)
    with args.data.open("rb") as handle:
        data = pickle.load(handle)
    training = prepare_split(data["train"])
    normalizers = {name: training[1].pop(name + "_stats") for name in MODALITIES}
    validation = prepare_split(data["valid"], normalizers)
    validation_ids = data["valid"]["id"]
    del data
    model = Baseline(args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    print(f"train={len(training[2]['class'])} valid={len(validation[2]['class'])}; classes={torch.bincount(training[2]['class']).tolist()}", flush=True)
    best_score = float("-inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        for features, masks, targets in batches(*training, args.batch_size, device, True):
            optimizer.zero_grad(set_to_none=True)
            logits, regression = model(features, masks)
            loss = F.cross_entropy(logits, targets["class"]) + args.regression_weight * F.smooth_l1_loss(regression, targets["regression"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            steps += 1
        scores, predictions, strengths = evaluate(model, validation, args.batch_size, device)
        print(f"epoch={epoch} train_loss={running_loss / steps:.4f} valid={json.dumps(scores)}", flush=True)
        selection_score = scores["f1_macro"] - 0.1 * scores["mae"]
        if selection_score > best_score:
            best_score = selection_score
            torch.save({
                "model": model.state_dict(),
                "normalizers": normalizers,
                "config": vars(args) | {"data": str(args.data), "output": str(args.output)},
                "epoch": epoch,
                "validation_metrics": scores,
            }, args.output / "best.pt")
            with (args.output / "valid_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "class_true", "class_pred", "strength_true", "strength_pred"])
                writer.writerows(zip(validation_ids, validation[2]["class"].tolist(), predictions, validation[2]["regression"].tolist(), strengths))
            (args.output / "best_metrics.json").write_text(json.dumps({"epoch": epoch, **scores}, indent=2), encoding="utf-8")
    print(f"done: {args.output / 'best_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
