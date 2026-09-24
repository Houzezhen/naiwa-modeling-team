from __future__ import annotations

import argparse
import csv
import json
import pathlib
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

pathlib.PosixPath = pathlib.WindowsPath
try:
    import numpy.core as _numpy_core
    sys.modules.setdefault("numpy._core", _numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", _numpy_core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", _numpy_core.numeric)
except Exception:
    pass

from train_multiview import FusionModel, MODALITIES


CLASSES = ("Negative", "Neutral", "Positive")
INPUT_DIMS = {"text": 768, "audio": 74, "vision": 35}


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    model = FusionModel(
        int(config["hidden"]),
        int(config["layers"]),
        float(config["dropout"]),
        bool(config["bidirectional"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def resample_to_50(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    valid = np.any(np.isfinite(values) & (values != 0), axis=-1)
    valid_positions = np.flatnonzero(valid)
    if not len(valid_positions):
        return np.zeros((50, values.shape[-1]), dtype=np.float32), np.zeros(50, dtype=bool)
    length = int(valid_positions[-1] + 1)
    source = np.nan_to_num(values[:length], nan=0.0, posinf=0.0, neginf=0.0)
    result = np.zeros((50, source.shape[-1]), dtype=np.float32)
    result_mask = np.zeros(50, dtype=bool)
    for target_index in range(50):
        left = target_index * length / 50.0
        right = (target_index + 1) * length / 50.0
        first = max(0, int(np.floor(left)))
        last = min(length - 1, int(np.ceil(right)) - 1)
        weighted_sum = np.zeros(source.shape[-1], dtype=np.float64)
        weight_total = 0.0
        for source_index in range(first, last + 1):
            overlap = max(0.0, min(right, source_index + 1.0) - max(left, source_index))
            if overlap > 0 and valid[source_index]:
                weighted_sum += overlap * source[source_index]
                weight_total += overlap
        if weight_total > 0:
            result[target_index] = (weighted_sum / weight_total).astype(np.float32)
            result_mask[target_index] = True
    return result, result_mask


def normalize(values: np.ndarray, mask: np.ndarray, name: str, normalizers: dict):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean, std = normalizers[name]
    normalized = np.clip((values - mean) / std, -10.0, 10.0)
    normalized *= mask[..., None]
    return torch.from_numpy(normalized), torch.from_numpy(mask.astype(bool))


def encode_text(text_source: dict, tokenizer, text_model, device: torch.device):
    if "text_bert" in text_source:
        token_data = np.asarray(text_source["text_bert"])
        input_ids = torch.from_numpy(token_data[:, 0, :].astype(np.int64)).to(device)
        attention_mask = torch.from_numpy(token_data[:, 1, :].astype(np.int64)).to(device)
        source_name = "provided_text_bert_token_ids"
    else:
        raw_text = str(np.asarray(text_source["raw_text"]).reshape(-1)[0])
        encoded = tokenizer(
            [raw_text],
            padding="max_length",
            truncation=True,
            max_length=50,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        source_name = "tokenized_raw_text"
    with torch.no_grad():
        output = text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = torch.stack(output.hidden_states[-4:], dim=0).mean(dim=0).cpu().numpy().astype(np.float32)
    text_values = np.zeros((1, 50, INPUT_DIMS["text"]), dtype=np.float32)
    text_mask = np.zeros((1, 50), dtype=bool)
    token_count = min(hidden.shape[1], 50)
    text_values[0, :token_count] = hidden[0, :token_count]
    text_mask[0, :token_count] = attention_mask[0, :token_count].cpu().numpy().astype(bool)
    return text_values, text_mask, source_name


def load_sample(path: Path, variant: str, normalizers: dict, text_mode: str, tokenizer, text_model, device: torch.device):
    source = pickle.loads(path.read_bytes())["test"]
    features, masks = {}, {}
    if text_mode == "distilbert":
        text_values, text_mask, text_source = encode_text(source, tokenizer, text_model, device)
    else:
        text_values = np.zeros((1, 50, INPUT_DIMS["text"]), dtype=np.float32)
        text_mask = np.zeros((1, 50), dtype=bool)
        text_source = "explicit_zero_mask_control"
    features["text"], masks["text"] = normalize(text_values, text_mask, "text", normalizers)
    for name in ("audio", "vision"):
        raw = np.asarray(source[name])
        if variant == "aligned":
            values = raw.astype(np.float32)
            mask = np.any(np.isfinite(values) & (values != 0), axis=-1)
        else:
            values, mask = resample_to_50(raw[0])
            values = values[None, ...]
            mask = mask[None, ...]
        features[name], masks[name] = normalize(values, mask, name, normalizers)
    return features, masks, text_source


def infer(model, features, masks, device):
    features = {name: value.to(device) for name, value in features.items()}
    masks = {name: value.to(device) for name, value in masks.items()}
    with torch.no_grad():
        logits, regression, _ = model(features, masks)
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    class_index = int(np.argmax(probabilities))
    return probabilities, float(regression[0].cpu().item()), class_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed Q2 model on attachment 3 samples.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--attachment3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-model", default="distilbert-base-uncased")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    tokenizer = text_model = None
    try:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.text_model, revision="12040accade4e8a0f71eabdb258fecc2e7e948be", use_fast=True)
        text_model = AutoModel.from_pretrained(args.text_model, revision="12040accade4e8a0f71eabdb258fecc2e7e948be").to(device).eval()
    except Exception as error:
        raise RuntimeError("DistilBERT is required for the attachment 3 text-feature test. Install requirements-text.txt first.") from error
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in ("aligned", "unaligned"):
        variant_dir = args.attachment3 / ("对齐版本" if variant == "aligned" else "未对齐版本")
        for path in sorted(variant_dir.glob("*.pkl")):
            for text_mode in ("distilbert", "zero"):
                features, masks, text_source = load_sample(
                    path, variant, checkpoint["normalizers"], text_mode, tokenizer, text_model, device
                )
                probabilities, strength, class_index = infer(model, features, masks, device)
                valid_counts = {name: int(masks[name][0].sum().item()) for name in MODALITIES}
                rows.append({
                    "variant": variant,
                    "text_mode": text_mode,
                    "file": path.name,
                    "id": path.stem,
                    "text_source": text_source,
                    "available_modalities": ",".join(name for name in MODALITIES if valid_counts[name] > 0),
                    "text_valid_bins": valid_counts["text"],
                    "audio_valid_bins": valid_counts["audio"],
                    "vision_valid_bins": valid_counts["vision"],
                    "predicted_class": CLASSES[class_index],
                    "predicted_strength": strength,
                    "prob_negative": float(probabilities[0]),
                    "prob_neutral": float(probabilities[1]),
                    "prob_positive": float(probabilities[2]),
                })
                print(f"inferred {variant} {path.name} text={text_mode}", flush=True)

    csv_path = args.output / "附件3_预测结果.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "sample_count": len(rows),
        "samples_per_variant_and_text_mode": {f"{variant}_{text_mode}": sum(row["variant"] == variant and row["text_mode"] == text_mode for row in rows) for variant in ("aligned", "unaligned") for text_mode in ("distilbert", "zero")},
        "predicted_class_counts": {f"{variant}_{text_mode}": dict(Counter(row["predicted_class"] for row in rows if row["variant"] == variant and row["text_mode"] == text_mode)) for variant in ("aligned", "unaligned") for text_mode in ("distilbert", "zero")},
        "predicted_strength_mean": {f"{variant}_{text_mode}": float(np.mean([row["predicted_strength"] for row in rows if row["variant"] == variant and row["text_mode"] == text_mode])) for variant in ("aligned", "unaligned") for text_mode in ("distilbert", "zero")},
        "valid_bins_mean": {f"{variant}_{text_mode}": {name: float(np.mean([row[name + "_valid_bins"] for row in rows if row["variant"] == variant and row["text_mode"] == text_mode])) for name in MODALITIES} for variant in ("aligned", "unaligned") for text_mode in ("distilbert", "zero")},
        "metrics": None,
        "metrics_note": "Attachment 3 has no ground-truth labels; accuracy, F1, MAE, and Pearson are not defined.",
        "text_note": "The primary distilbert mode reconstructs 768-dimensional features from supplied token IDs or raw text using the same DistilBERT last-four-layer mean convention as Q1; zero mode is retained as a strict missing-text sensitivity control.",
        "output_csv": str(csv_path),
    }
    (args.output / "附件3_测试汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
