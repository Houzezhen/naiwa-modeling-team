import argparse
import csv
import json
import pathlib
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

pathlib.PosixPath = pathlib.WindowsPath
try:
    import numpy.core as _numpy_core
    sys.modules.setdefault("numpy._core", _numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", _numpy_core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", _numpy_core.numeric)
except Exception:
    pass

Q2_DIR = Path(__file__).resolve().parents[1] / "q2_baseline"
sys.path.insert(0, str(Q2_DIR))
from train_multiview import FusionModel, MODALITIES, convert_split

CLASSES = ("Negative", "Neutral", "Positive")


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_bundle(checkpoint_path, device):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    config = checkpoint["config"]
    model = FusionModel(
        config["hidden"], config["layers"], config["dropout"], config["bidirectional"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def forward(model, features, masks, device):
    features = {name: value.to(device) for name, value in features.items()}
    masks = {name: value.to(device) for name, value in masks.items()}
    with torch.no_grad():
        logits, regression, _ = model(features, masks)
    return logits, regression


def metrics(truth, predicted, actual, estimate):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    actual, estimate = np.asarray(actual), np.asarray(estimate)
    f1_values = []
    for label in range(3):
        tp = np.sum((truth == label) & (predicted == label))
        fp = np.sum((truth != label) & (predicted == label))
        fn = np.sum((truth == label) & (predicted != label))
        denominator = 2 * tp + fp + fn
        f1_values.append(float(2 * tp / denominator) if denominator else 0.0)
    pearson = float(np.corrcoef(actual, estimate)[0, 1]) if np.std(actual) and np.std(estimate) else 0.0
    return {
        "accuracy": float(np.mean(truth == predicted)),
        "f1_macro": float(np.mean(f1_values)),
        "mae": float(np.mean(np.abs(actual - estimate))),
        "pearson": pearson,
    }


def evaluate(model, dataset, device, batch_size=128):
    features, masks, targets, _ = dataset
    predicted, truth, estimate, actual = [], [], [], []
    for start in range(0, len(targets["class"]), batch_size):
        positions = slice(start, start + batch_size)
        batch_features = {name: value[positions] for name, value in features.items()}
        batch_masks = {name: value[positions] for name, value in masks.items()}
        logits, regression = forward(model, batch_features, batch_masks, device)
        predicted.extend(logits.argmax(-1).cpu().tolist())
        truth.extend(targets["class"][positions].tolist())
        estimate.extend(regression.cpu().tolist())
        actual.extend(targets["regression"][positions].tolist())
    return metrics(truth, predicted, actual, estimate)


def hide_modality(features, masks, modality, index=None):
    hidden_features = {name: value.clone() for name, value in features.items()}
    hidden_masks = {name: value.clone() for name, value in masks.items()}
    if index is None:
        hidden_masks[modality] = torch.zeros_like(hidden_masks[modality])
        hidden_features[modality].zero_()
    else:
        hidden_masks[modality][:, index] = False
        hidden_features[modality][:, index] = 0.0
    return hidden_features, hidden_masks


def explain_one(model, features, masks, device, top_k=3, evidence_count=6):
    logits, regression = forward(model, features, masks, device)
    probabilities = F.softmax(logits, dim=-1)[0]
    class_index = int(logits.argmax(-1)[0].item())
    base_probability = float(probabilities[class_index].item())
    base_strength = float(regression[0].item())
    modality_drops, modality_regression_changes = {}, {}
    window_drops = {}
    for modality in MODALITIES:
        hidden_features, hidden_masks = hide_modality(features, masks, modality)
        hidden_logits, hidden_regression = forward(model, hidden_features, hidden_masks, device)
        modality_drops[modality] = max(base_probability - float(F.softmax(hidden_logits, dim=-1)[0, class_index].item()), 0.0)
        modality_regression_changes[modality] = abs(base_strength - float(hidden_regression[0].item()))
        count = features[modality].shape[1]
        batch_features = {name: value.repeat(count, 1, 1) for name, value in features.items()}
        batch_masks = {name: value.repeat(count, 1) for name, value in masks.items()}
        indices = torch.arange(count)
        batch_masks[modality][indices, indices] = False
        batch_features[modality][indices, indices] = 0.0
        hidden_logits, hidden_regression = forward(model, batch_features, batch_masks, device)
        hidden_probability = F.softmax(hidden_logits, dim=-1)[:, class_index].cpu().numpy()
        hidden_strength = hidden_regression.cpu().numpy()
        window_drops[modality] = [
            {
                "window_index": index,
                "occlusion_drop": max(base_probability - float(hidden_probability[index]), 0.0),
                "regression_change": abs(base_strength - float(hidden_strength[index])),
            }
            for index in range(count)
        ]
    total_drop = sum(modality_drops.values())
    if total_drop > 1e-8:
        contributions = {name: value / total_drop for name, value in modality_drops.items()}
    else:
        contributions = {name: 1.0 / len(MODALITIES) for name in MODALITIES}
    evidence = []
    for modality in MODALITIES:
        ranked = sorted(window_drops[modality], key=lambda item: item["occlusion_drop"], reverse=True)[:top_k]
        for item in ranked:
            evidence.append({
                "modality": modality,
                "window_index": item["window_index"],
                "occlusion_drop": item["occlusion_drop"],
                "modality_contribution": contributions[modality],
                "evidence_score": item["occlusion_drop"] * contributions[modality],
                "regression_change": item["regression_change"],
            })
    evidence.sort(key=lambda item: item["evidence_score"], reverse=True)
    main_modality = max(contributions, key=contributions.get)
    return {
        "class_index": class_index,
        "strength": base_strength,
        "main_modality": main_modality,
        "contributions": contributions,
        "modality_drops": modality_drops,
        "modality_regression_changes": modality_regression_changes,
        "window_drops": window_drops,
        "evidence": evidence[:evidence_count],
    }


def sample_to_tensors(sample, normalizers):
    features, masks = {}, {}
    for name in MODALITIES:
        values = np.asarray(sample[name], dtype=np.float32)[None, ...]
        if name == "text":
            mask = np.asarray(sample["text_bert"])[None, 1, :] != 0
        else:
            mask = np.any(np.isfinite(values) & (values != 0), axis=-1)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mean, std = normalizers[name]
        values = np.clip((values - mean) / std, -10.0, 10.0) * mask[..., None]
        features[name] = torch.from_numpy(values)
        masks[name] = torch.from_numpy(mask)
    return features, masks


def text_fragment(text, index):
    words = str(text).split()
    if not words:
        return ""
    start = int(len(words) * index / 50)
    end = max(start + 1, int(len(words) * (index + 1) / 50))
    return " ".join(words[start:end])


def video_duration(video_path):
    try:
        import av
        container = av.open(str(video_path))
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        container.close()
        return duration
    except Exception:
        return 0.0


def export_frame(video_path, sample_id, index, duration, output_dir):
    try:
        import av
        container = av.open(str(video_path))
        target = duration * (index + 0.5) / 50.0
        container.seek(int(target * av.time_base), any_frame=False, backward=True)
        image = None
        for frame in container.decode(video=0):
            image = frame.to_image()
            break
        container.close()
        if image is None:
            return ""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{sample_id}_window_{index:02d}.jpg"
        image.save(path, quality=90)
        return str(path)
    except Exception:
        return ""


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_explanations(model, dataset, device, samples, seed):
    features, masks, _, ids = dataset
    rng = random.Random(seed)
    selected = list(range(min(samples, len(ids))))
    rng.shuffle(selected)
    selected = selected[:samples]
    top_drops, random_drops = [], []
    for sample_index in selected:
        one_features = {name: value[sample_index:sample_index + 1] for name, value in features.items()}
        one_masks = {name: value[sample_index:sample_index + 1] for name, value in masks.items()}
        result = explain_one(model, one_features, one_masks, device, top_k=1, evidence_count=1)
        for modality in MODALITIES:
            window_effects = [item["occlusion_drop"] for item in result["window_drops"][modality]]
            top_index = int(np.argmax(window_effects))
            random_index = rng.randrange(one_features[modality].shape[1])
            top_drops.append(max(window_effects[top_index], 0.0))
            random_drops.append(max(window_effects[random_index], 0.0))
    return {
        "samples": len(selected),
        "mean_top_window_drop": float(np.mean(top_drops)),
        "mean_random_window_drop": float(np.mean(random_drops)),
        "top_over_random": float(np.mean(top_drops) - np.mean(random_drops)),
    }


def infer_attachment4(model, checkpoint, attachment4, output, device, top_k, evidence_count):
    root = Path(attachment4) / "对齐版本"
    rows, evidence_rows = [], []
    for path in sorted(root.glob("*.pkl")):
        sample = load_pickle(path)
        features, masks = sample_to_tensors(sample, checkpoint["normalizers"])
        result = explain_one(model, features, masks, device, top_k, evidence_count)
        sample_id = str(sample.get("id", path.stem))
        rows.append({
            "id": sample_id,
            "predicted_class": CLASSES[result["class_index"]],
            "predicted_strength": result["strength"],
            "main_modality": result["main_modality"],
            **{name + "_contribution": value for name, value in result["contributions"].items()},
            **{name + "_occlusion_drop": value for name, value in result["modality_drops"].items()},
        })
        duration = video_duration(root / "videos" / f"{path.stem}.mp4")
        effective_duration = duration if duration > 0 else 1.0
        frame_dir = output / "evidence_frames"
        for rank, item in enumerate(result["evidence"], 1):
            evidence_rows.append({
                "id": sample_id,
                "rank": rank,
                **item,
                "start_time": effective_duration * item["window_index"] / 50,
                "end_time": effective_duration * (item["window_index"] + 1) / 50,
                "start_time_normalized": item["window_index"] / 50,
                "end_time_normalized": (item["window_index"] + 1) / 50,
                "text_fragment": text_fragment(sample.get("raw_text", ""), item["window_index"]),
                "frame_path": export_frame(root / "videos" / f"{path.stem}.mp4", sample_id, item["window_index"], effective_duration, frame_dir) if item["modality"] == "vision" else "",
            })
        print("inferred", sample_id, flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "q3_predictions.csv", rows)
    write_csv(output / "q3_explanations.csv", evidence_rows)
    return {"samples": len(rows), "evidence_rows": len(evidence_rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--attachment4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--evidence-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model, checkpoint = load_bundle(args.checkpoint, args.device)
    source = load_pickle(args.data)
    valid = convert_split(source["valid"], "aligned", checkpoint["normalizers"])
    validation_metrics = evaluate(model, valid, args.device, args.batch_size)
    explanation_metrics = validate_explanations(model, valid, args.device, args.validation_samples, args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "q2_backbone_validation.json").write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
    (output / "q2_backbone_explanation_validation.json").write_text(json.dumps({**validation_metrics, **explanation_metrics}, indent=2), encoding="utf-8")
    attachment_result = infer_attachment4(model, checkpoint, args.attachment4, output / "attachment4", args.device, args.top_k, args.evidence_count)
    print(json.dumps({"validation": validation_metrics, "explanation": explanation_metrics, "attachment4": attachment_result}), flush=True)


if __name__ == "__main__":
    main()
