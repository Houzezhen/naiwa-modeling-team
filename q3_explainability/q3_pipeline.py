import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODALITIES = ("text", "audio", "vision")
DIMS = {"text": 768, "audio": 74, "vision": 35}
CLASSES = ("Negative", "Neutral", "Positive")

try:
    import numpy.core as _numpy_core
    sys.modules.setdefault("numpy._core", _numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", _numpy_core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", _numpy_core.numeric)
except Exception:
    pass


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def get_mask(split, name):
    values = np.asarray(split[name])
    if name == "text":
        return np.asarray(split["text_bert"])[:, 1, :] != 0
    if name + "_mask" in split:
        return np.asarray(split[name + "_mask"], dtype=bool)
    return np.any(np.isfinite(values) & (values != 0), axis=-1)


def normalizers(split):
    result = {}
    for name in MODALITIES:
        values = np.nan_to_num(np.asarray(split[name], dtype=np.float32))
        present = values[get_mask(split, name)]
        result[name] = (present.mean(0).astype(np.float32), np.maximum(present.std(0), 1e-4).astype(np.float32))
    return result


def convert(split, stats, labeled=True):
    features, masks = {}, {}
    for name in MODALITIES:
        values = np.nan_to_num(np.asarray(split[name], dtype=np.float32))
        mask = get_mask(split, name)
        mean, std = stats[name]
        values = np.clip((values - mean) / std, -10, 10) * mask[..., None]
        features[name], masks[name] = torch.from_numpy(values), torch.from_numpy(mask)
    targets = None
    if labeled:
        targets = {"class": torch.from_numpy(np.asarray(split["classification_labels"], dtype=np.int64)), "regression": torch.from_numpy(np.asarray(split["regression_labels"], dtype=np.float32))}
    return features, masks, targets, list(split.get("id", []))


def make_batches(dataset, size, device, shuffle, seed=0):
    features, masks, targets, _ = dataset
    count = len(next(iter(features.values())))
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randperm(count, generator=generator) if shuffle else torch.arange(count)
    for start in range(0, count, size):
        current = positions[start:start + size]
        yield ({name: features[name][current].to(device) for name in MODALITIES}, {name: masks[name][current].to(device) for name in MODALITIES}, None if targets is None else {name: value[current].to(device) for name, value in targets.items()})


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout, bidirectional):
        super().__init__()
        if bidirectional and hidden % 2:
            raise ValueError("hidden must be even for bidirectional GRU")
        self.projection = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(hidden, hidden // 2 if bidirectional else hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0, bidirectional=bidirectional)
        self.norm, self.dropout = nn.LayerNorm(hidden), nn.Dropout(dropout)

    def forward(self, values, mask):
        encoded = F.gelu(self.projection(values)) * mask.unsqueeze(-1)
        temporal, _ = self.gru(encoded)
        return self.norm(encoded + self.dropout(temporal)) * mask.unsqueeze(-1)


class ExplainableModel(nn.Module):
    def __init__(self, hidden=128, layers=2, dropout=0.15, bidirectional=True):
        super().__init__()
        self.encoders = nn.ModuleDict({name: Encoder(DIMS[name], hidden, layers, dropout, bidirectional) for name in MODALITIES})
        self.time_score = nn.ModuleDict({name: nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1)) for name in MODALITIES})
        self.gate = nn.Sequential(nn.LayerNorm(hidden * 3), nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.classifier, self.regressor = nn.Linear(hidden, 3), nn.Linear(hidden, 1)

    def encode(self, features, masks):
        pooled, attentions = [], []
        for name in MODALITIES:
            tokens = self.encoders[name](features[name], masks[name])
            scores = self.time_score[name](tokens).squeeze(-1).masked_fill(~masks[name], -1e4)
            attention = F.softmax(scores, 1) * masks[name].float()
            attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-6)
            pooled.append((tokens * attention.unsqueeze(-1)).sum(1))
            attentions.append(attention)
        stacked = torch.stack(pooled, 1)
        contribution = F.softmax(self.gate(stacked.flatten(1)), -1)
        fused = self.fusion((stacked * contribution.unsqueeze(-1)).flatten(1))
        return fused, contribution, attentions

    def forward(self, features, masks):
        fused, contribution, attentions = self.encode(features, masks)
        return self.classifier(fused), self.regressor(fused).squeeze(-1), contribution, attentions


def score_metrics(truth, predicted, actual, estimate):
    truth, predicted, actual, estimate = map(np.asarray, (truth, predicted, actual, estimate))
    f1 = []
    for label in range(3):
        tp = np.sum((truth == label) & (predicted == label))
        fp = np.sum((truth != label) & (predicted == label))
        fn = np.sum((truth == label) & (predicted != label))
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    pearson = np.corrcoef(actual, estimate)[0, 1] if np.std(actual) and np.std(estimate) else 0.0
    return {"accuracy": float(np.mean(truth == predicted)), "f1_macro": float(np.mean(f1)), "mae": float(np.mean(np.abs(actual - estimate))), "pearson": float(pearson)}


@torch.no_grad()
def evaluate(model, dataset, batch_size, device):
    model.eval()
    truth, predicted, actual, estimate = [], [], [], []
    for features, masks, targets in make_batches(dataset, batch_size, device, False):
        logits, regression, _, _ = model(features, masks)
        truth += targets["class"].cpu().tolist()
        predicted += logits.argmax(-1).cpu().tolist()
        actual += targets["regression"].cpu().tolist()
        estimate += regression.cpu().tolist()
    return score_metrics(truth, predicted, actual, estimate)


def load_q2_encoders(model, path):
    if not path or not Path(path).exists():
        return {"loaded": False}
    state = torch.load(str(path), map_location="cpu")
    state = state.get("model", state)
    encoders = {key[9:]: value for key, value in state.items() if key.startswith("encoders.")}
    result = model.encoders.load_state_dict(encoders, strict=False)
    return {"loaded": True, "missing": result.missing_keys, "unexpected": result.unexpected_keys}


def train(args):
    seed_everything(args.seed)
    source = load_pickle(args.data)
    stats = normalizers(source["train"])
    train_set, valid_set = convert(source["train"], stats), convert(source["valid"], stats)
    config = {"hidden": args.hidden, "layers": args.layers, "dropout": args.dropout, "bidirectional": args.bidirectional}
    model = ExplainableModel(**config).to(args.device)
    initialization = load_q2_encoders(model, args.init_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_score, best_state, history = -1e9, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, masks, targets in make_batches(train_set, args.batch_size, args.device, True, args.seed + epoch):
            logits, regression, _, _ = model(features, masks)
            loss = F.cross_entropy(logits, targets["class"]) + args.regression_weight * F.smooth_l1_loss(regression, targets["regression"])
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(loss.item())
        validation = evaluate(model, valid_set, args.batch_size, args.device)
        score = validation["f1_macro"] - 0.1 * validation["mae"]
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **validation}
        history.append(record); print(json.dumps(record), flush=True)
        if score > best_score:
            best_score, best_state = score, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state); final = evaluate(model, valid_set, args.batch_size, args.device)
    checkpoint = {"model": best_state, "normalizers": stats, "config": config, "initialization": initialization, "validation": final, "history": history}
    torch.save(checkpoint, output / "q3_explainable_best.pt")
    (output / "validation.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"validation": final, "initialization": initialization}), flush=True)


def load_model(path, device):
    checkpoint = torch.load(str(path), map_location="cpu")
    model = ExplainableModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    return model, checkpoint


def one_sample(sample, stats):
    features, masks = {}, {}
    for name in MODALITIES:
        raw = np.asarray(sample[name], dtype=np.float32)[None]
        mask = np.asarray(sample["text_bert"])[None, 1, :] != 0 if name == "text" else np.any(np.isfinite(raw) & (raw != 0), -1)
        mean, std = stats[name]
        values = np.clip((np.nan_to_num(raw) - mean) / std, -10, 10) * mask[..., None]
        features[name], masks[name] = torch.from_numpy(values), torch.from_numpy(mask)
    return features, masks


def hide_window(features, masks, modality, index):
    new_features = {name: value.clone() for name, value in features.items()}
    new_masks = {name: value.clone() for name, value in masks.items()}
    new_features[modality][:, index] = 0; new_masks[modality][:, index] = False
    return new_features, new_masks


def explain_sample(model, features, masks, device, top_k):
    features = {name: value.to(device) for name, value in features.items()}; masks = {name: value.to(device) for name, value in masks.items()}
    with torch.no_grad():
        logits, regression, contribution, attentions = model(features, masks)
        probability = F.softmax(logits, -1)[0]; class_index = int(logits.argmax(-1)[0]); base_probability = probability[class_index].item(); base_regression = regression[0].item()
        contribution = contribution[0].cpu().numpy(); attentions = {name: attentions[i][0].cpu().numpy() for i, name in enumerate(MODALITIES)}
    evidence = []
    for modality_index, modality in enumerate(MODALITIES):
        for index in np.argsort(-attentions[modality])[:top_k]:
            changed_features, changed_masks = hide_window(features, masks, modality, int(index))
            with torch.no_grad():
                changed_logits, changed_regression, _, _ = model(changed_features, changed_masks)
            changed_probability = F.softmax(changed_logits, -1)[0, class_index].item()
            drop = max(base_probability - changed_probability, 0.0)
            evidence.append({"modality": modality, "window_index": int(index), "attention_score": float(attentions[modality][index]), "modality_contribution": float(contribution[modality_index]), "occlusion_drop": float(drop), "evidence_score": float(attentions[modality][index] * contribution[modality_index] * drop), "regression_change": float(abs(base_regression - changed_regression[0].item()))})
    evidence.sort(key=lambda item: item["evidence_score"], reverse=True)
    return {"class_index": class_index, "strength": float(base_regression), "main_modality": MODALITIES[int(np.argmax(contribution))], "contributions": {name: float(contribution[i]) for i, name in enumerate(MODALITIES)}, "evidence": evidence}


def text_fragment(text, index):
    words = str(text).split(); start = int(len(words) * index / 50); end = max(start + 1, int(len(words) * (index + 1) / 50))
    return " ".join(words[start:end])


def frame(video, sample_id, index, duration, output_dir):
    try:
        import cv2
    except Exception:
        return ""
    capture = cv2.VideoCapture(str(video)); capture.set(cv2.CAP_PROP_POS_MSEC, duration * index / 50 * 1000); ok, image = capture.read(); capture.release()
    if not ok:
        return ""
    output_dir.mkdir(parents=True, exist_ok=True); path = output_dir / f"{sample_id}_window_{index:02d}.jpg"; cv2.imwrite(str(path), image); return str(path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def infer(args):
    model, checkpoint = load_model(args.checkpoint, args.device); root = Path(args.attachment4); feature_dir = root / "对齐版本"; video_dir = feature_dir / "videos"; output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    prediction_rows, evidence_rows = [], []
    for feature_path in sorted(feature_dir.glob("*.pkl")):
        sample = load_pickle(feature_path); result = explain_sample(model, *one_sample(sample, checkpoint["normalizers"]), args.device, args.top_k); sample_id = str(sample.get("id", feature_path.stem))
        prediction_rows.append({"id": sample_id, "predicted_class": CLASSES[result["class_index"]], "predicted_strength": result["strength"], "main_modality": result["main_modality"], **{name + "_contribution": value for name, value in result["contributions"].items()}})
        video_path = video_dir / (feature_path.stem + ".mp4")
        duration = 1.0
        for rank, item in enumerate(result["evidence"][:args.evidence_count], 1):
            evidence_rows.append({"id": sample_id, "rank": rank, **item, "start_time": duration * item["window_index"] / 50, "end_time": duration * (item["window_index"] + 1) / 50, "text_fragment": text_fragment(sample.get("raw_text", ""), item["window_index"]), "frame_path": ""})
        print("inferred", sample_id, flush=True)
    write_csv(output / "q3_predictions.csv", prediction_rows); write_csv(output / "q3_explanations.csv", evidence_rows)


def explanation_validation(args):
    model, checkpoint = load_model(args.checkpoint, args.device); source = load_pickle(args.data); valid = convert(source["valid"], checkpoint["normalizers"]); metrics_result = evaluate(model, valid, args.batch_size, args.device); rng = np.random.RandomState(args.seed); count = min(args.samples, len(valid[3])); indices = rng.choice(len(valid[3]), count, replace=False); top_drops, random_drops = [], []
    for sample_index in indices:
        features = {name: valid[0][name][sample_index:sample_index + 1] for name in MODALITIES}; masks = {name: valid[1][name][sample_index:sample_index + 1] for name in MODALITIES}
        with torch.no_grad():
            logits, _, _, attentions = model({name: value.to(args.device) for name, value in features.items()}, {name: value.to(args.device) for name, value in masks.items()}); class_index = int(logits.argmax(-1)[0]); base = F.softmax(logits, -1)[0, class_index].item()
        for modality_index, modality in enumerate(MODALITIES):
            top = int(attentions[modality_index][0].argmax()); random_index = int(rng.randint(0, 50)); drops = []
            for index in (top, random_index):
                changed_features, changed_masks = hide_window(features, masks, modality, index)
                with torch.no_grad():
                    changed_logits, _, _, _ = model({name: value.to(args.device) for name, value in changed_features.items()}, {name: value.to(args.device) for name, value in changed_masks.items()})
                drops.append(base - F.softmax(changed_logits, -1)[0, class_index].item())
            top_drops.append(max(drops[0], 0)); random_drops.append(max(drops[1], 0))
    result = {"validation_metrics": metrics_result, "samples": count, "mean_top_attention_drop": float(np.mean(top_drops)), "mean_random_drop": float(np.mean(random_drops)), "top_over_random": float(np.mean(top_drops) - np.mean(random_drops))}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train"); train_parser.add_argument("--data", type=Path, required=True); train_parser.add_argument("--init-checkpoint", type=Path); train_parser.add_argument("--output", type=Path, required=True); train_parser.add_argument("--epochs", type=int, default=15); train_parser.add_argument("--batch-size", type=int, default=64); train_parser.add_argument("--hidden", type=int, default=128); train_parser.add_argument("--layers", type=int, default=2); train_parser.add_argument("--dropout", type=float, default=0.15); train_parser.add_argument("--learning-rate", type=float, default=2e-4); train_parser.add_argument("--regression-weight", type=float, default=0.5); train_parser.add_argument("--bidirectional", action="store_true"); train_parser.add_argument("--seed", type=int, default=42); train_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu"); train_parser.set_defaults(function=train)
    valid_parser = sub.add_parser("validate"); valid_parser.add_argument("--data", type=Path, required=True); valid_parser.add_argument("--checkpoint", type=Path, required=True); valid_parser.add_argument("--output", type=Path, required=True); valid_parser.add_argument("--batch-size", type=int, default=64); valid_parser.add_argument("--samples", type=int, default=128); valid_parser.add_argument("--seed", type=int, default=42); valid_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu"); valid_parser.set_defaults(function=explanation_validation)
    infer_parser = sub.add_parser("infer-attachment4"); infer_parser.add_argument("--attachment4", type=Path, required=True); infer_parser.add_argument("--checkpoint", type=Path, required=True); infer_parser.add_argument("--output", type=Path, required=True); infer_parser.add_argument("--top-k", type=int, default=3); infer_parser.add_argument("--evidence-count", type=int, default=6); infer_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu"); infer_parser.set_defaults(function=infer)
    args = parser.parse_args(); args.function(args)


if __name__ == "__main__":
    main()
