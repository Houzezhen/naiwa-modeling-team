"""Train and infer with one aligned version and one text_bert interface."""

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from text_bert_features import FEATURE_RULE, MODEL_NAME, MODEL_REVISION, encode_text_bert, load_encoder
from train_multiview import MODALITIES, load_view, set_seed, train_separate_fusion, train_single_modality
from evaluate_attachment3 import CLASSES, infer, load_checkpoint, normalize


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def prepare(args, device):
    with args.aligned_data.open("rb") as handle:
        source = pickle.load(handle)
    encoder = load_encoder(device)
    derived = {}
    for name in ("train", "valid", "test"):
        split = source[name]
        derived[name] = {key: split[key] for key in ("text_bert", "id", "classification_labels", "regression_labels")}
        for modality in ("audio", "vision"):
            derived[name][modality] = np.asarray(split[modality], dtype=np.float32)
        derived[name]["text"] = encode_text_bert(split["text_bert"], encoder, device, args.text_batch_size)
        print(f"encoded {name}: {len(derived[name]['text'])}", flush=True)
        del source[name]
    args.derived_data.parent.mkdir(parents=True, exist_ok=True)
    with args.derived_data.open("wb") as handle:
        pickle.dump(derived, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {"input": str(args.aligned_data), "input_sha256": digest(args.aligned_data),
                "derived": str(args.derived_data), "derived_sha256": digest(args.derived_data),
                "model": MODEL_NAME, "revision": MODEL_REVISION, "feature_rule": FEATURE_RULE,
                "feature_version": "aligned_50", "text_interface": "text_bert[N,3,50]",
                "splits": {key: len(value["id"]) for key, value in derived.items()}}
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def check_manifest(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["input_sha256"] != digest(args.aligned_data) or manifest["derived_sha256"] != digest(args.derived_data):
        raise ValueError("Data SHA256 differs from preparation manifest")


def train(args, device):
    check_manifest(args)
    set_seed(args.seed)
    view = load_view(args.derived_data, "aligned")
    args.output.mkdir(parents=True, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    for modality in MODALITIES:
        train_single_modality(modality, view, args.output, args, device, scaler)
    train_separate_fusion(view, args.output, args, device, scaler)
    (args.output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")


def predict(args, device):
    check_manifest(args)
    model, checkpoint = load_checkpoint(args.output / "fusion_best.pt", device)
    encoder = load_encoder(device)
    rows = []
    files = sorted((args.attachment3 / "对齐版本").glob("*.pkl"))
    if len(files) != 30:
        raise ValueError(f"Expected 30 aligned attachment 3 files, got {len(files)}")
    for path in files:
        with path.open("rb") as handle:
            source = pickle.load(handle)["test"]
        features, masks = {}, {}
        text = encode_text_bert(source["text_bert"], encoder, device, args.text_batch_size)
        text_mask = np.asarray(source["text_bert"])[:, 1, :] != 0
        features["text"], masks["text"] = normalize(text, text_mask, "text", checkpoint["normalizers"])
        for modality in ("audio", "vision"):
            values = np.asarray(source[modality], dtype=np.float32)
            mask = np.any(np.isfinite(values) & (values != 0), axis=-1)
            features[modality], masks[modality] = normalize(values, mask, modality, checkpoint["normalizers"])
        probabilities, strength, prediction = infer(model, features, masks, device)
        rows.append({"file": path.name, "predicted_class": CLASSES[prediction],
                     "predicted_strength": strength,
                     **{f"prob_{name.lower()}": float(probabilities[index]) for index, name in enumerate(CLASSES)},
                     **{f"{name}_valid_bins": int(masks[name].sum().item()) for name in MODALITIES}})
        print(f"inferred {path.name}", flush=True)
    predictions = args.output / "附件3_统一编码预测结果.csv"
    with predictions.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"feature_version": "aligned_50", "text_input": "text_bert (3x50)",
               "text_model": MODEL_NAME, "revision": MODEL_REVISION, "feature_rule": FEATURE_RULE,
               "checkpoint": str(args.output / "fusion_best.pt"), "checkpoint_sha256": digest(args.output / "fusion_best.pt"),
               "sample_count": len(rows), "class_counts": dict(Counter(row["predicted_class"] for row in rows)),
               "mean_strength": float(np.mean([row["predicted_strength"] for row in rows])),
               "metrics": None, "metrics_note": "Attachment 3 has no labels; accuracy/F1/MAE/Pearson undefined.",
               "prediction_csv": str(predictions)}
    (args.output / "附件3_统一编码测试汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "train", "predict"))
    parser.add_argument("--aligned-data", type=Path, default=Path("emotion-transfer-pilot/aligned_50.pkl"))
    parser.add_argument("--derived-data", type=Path, default=Path("emotion-transfer-pilot/aligned_text_bert_features.pkl"))
    parser.add_argument("--manifest", type=Path, default=Path("q2_baseline/aligned_text_bert_manifest.json"))
    parser.add_argument("--attachment3", type=Path, default=Path("E题复杂场景下多模态情感识别的数学建模与算法设计/E题数据/E题数据/附件3-模态缺失特征样本"))
    parser.add_argument("--output", type=Path, default=Path("q2_baseline/aligned_text_bert_run"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--projector-hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoder-epochs", type=int, default=3)
    parser.add_argument("--fusion-epochs", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--encoder-learning-rate", type=float, default=3e-4)
    parser.add_argument("--fusion-learning-rate", type=float, default=2e-4)
    parser.add_argument("--fusion-encoder-learning-rate", type=float, default=5e-5)
    parser.add_argument("--regression-weight", type=float, default=0.5)
    parser.add_argument("--feature-noise", type=float, default=0.05)
    parser.add_argument("--time-mask-probability", type=float, default=0.15)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--covariance-weight", type=float, default=0.04)
    parser.add_argument("--variance-floor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.step == "prepare":
        prepare(args, device)
    elif args.step == "train":
        train(args, device)
    else:
        predict(args, device)


if __name__ == "__main__":
    main()
