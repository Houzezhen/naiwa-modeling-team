from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_typical_sample(base: dict) -> int:
    durations = np.asarray(base["durations"])
    median_duration = float(np.median(durations))
    audio_coverage = np.asarray(base["audio_mask"]).mean(axis=1)
    vision_coverage = np.asarray(base["vision_mask"]).mean(axis=1)
    duration_distance = np.abs(durations - median_duration) / max(median_duration, 1e-6)
    score = audio_coverage + vision_coverage - 0.25 * duration_distance
    return int(np.argmax(score))


def load_text_alignment(path: Path) -> dict[str, dict]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[record["sample_id"]] = record
    return records


def write_typical_csv(
    output_path: Path,
    base: dict,
    text_mask: np.ndarray,
    text_alignment: dict[str, dict],
    sample_index: int,
) -> None:
    sample_id = base["id"][sample_index]
    alignment = text_alignment[sample_id]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "sample_id",
            "bin_index",
            "start_s",
            "end_s",
            "text_tokens",
            "text_valid",
            "audio_valid",
            "audio_rms_mean",
            "vision_valid",
            "vision_timestamp_s",
            "face_detected",
            "visual_motion_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bin_index, bounds in enumerate(base["time_bounds"][sample_index]):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "bin_index": bin_index,
                    "start_s": f"{float(bounds[0]):.6f}",
                    "end_s": f"{float(bounds[1]):.6f}",
                    "text_tokens": " ".join(alignment["bin_tokens"][bin_index]),
                    "text_valid": int(text_mask[sample_index, bin_index]),
                    "audio_valid": int(base["audio_mask"][sample_index, bin_index]),
                    "audio_rms_mean": f"{float(base['audio'][sample_index, bin_index, 72]):.8f}",
                    "vision_valid": int(base["vision_mask"][sample_index, bin_index]),
                    "vision_timestamp_s": (
                        f"{float(base['vision_timestamps'][sample_index, bin_index]):.6f}"
                        if np.isfinite(base["vision_timestamps"][sample_index, bin_index])
                        else ""
                    ),
                    "face_detected": int(base["vision"][sample_index, bin_index, 0] >= 0.5),
                    "visual_motion_mean": f"{float(base['vision'][sample_index, bin_index, 25]):.8f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble final question 1 feature and audit files.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.work_dir / "base_features.pkl").open("rb") as handle:
        base = pickle.load(handle)
    text = np.load(args.work_dir / "text_features.npy")
    text_mask = np.load(args.work_dir / "text_mask.npy")
    text_model = json.loads((args.work_dir / "text_model.json").read_text(encoding="utf-8"))
    text_alignment = load_text_alignment(args.work_dir / "text_alignment.jsonl")

    all_data = {
        "raw_text": np.asarray(base["raw_text"]),
        "audio": np.asarray(base["audio"], dtype=np.float32),
        "vision": np.asarray(base["vision"], dtype=np.float32),
        "text": np.asarray(text, dtype=np.float32),
        "id": list(base["id"]),
        "video_id": np.asarray(base["video_id"]),
        "clip_id": np.asarray(base["clip_id"]),
        "annotations": np.asarray(base["annotations"]),
        "classification_labels": np.asarray(base["classification_labels"], dtype=np.int64),
        "regression_labels": np.asarray(base["regression_labels"], dtype=np.float32),
        "time_bounds": np.asarray(base["time_bounds"], dtype=np.float32),
        "text_mask": np.asarray(text_mask, dtype=np.uint8),
        "audio_mask": np.asarray(base["audio_mask"], dtype=np.uint8),
        "vision_mask": np.asarray(base["vision_mask"], dtype=np.uint8),
        "vision_timestamps": np.asarray(base["vision_timestamps"], dtype=np.float32),
        "durations": np.asarray(base["durations"], dtype=np.float32),
        "source_file": list(base["source_file"]),
    }
    payload = {
        "all": all_data,
        "metadata": {
            "schema_version": "question1-aligned-50-v1",
            "alignment_rule": "50 equal-duration half-open bins [kT/50,(k+1)T/50); the final bin includes T",
            "padding_rule": "all-zero vector only when the matching modality mask is zero",
            "classification_mapping": {"Negative": 0, "Neutral": 1, "Positive": 2},
            "feature_dimensions": {"text": 768, "audio": 74, "vision": 35},
            "text_extractor": text_model,
            "audio_extractor": "16 kHz mono; 20 MFCC mean/std, 12 chroma means, 8 spectral descriptors mean/std, F0/ZCR/RMS mean/std",
            "vision_extractor": "OpenCV Haar face geometry, intensity/color/texture, face-region symmetry, dense optical flow",
            "software": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
        },
    }
    output_pkl = args.output_dir / "question1_aligned_50.pkl"
    reconstruct = np._core.multiarray._reconstruct
    reconstruct_module = reconstruct.__module__
    reconstruct.__module__ = "numpy.core.multiarray"
    try:
        with output_pkl.open("wb") as handle:
            pickle.dump(payload, handle, protocol=4)
    finally:
        reconstruct.__module__ = reconstruct_module

    for source_name in ["feature_manifest.csv", "processing_log.jsonl", "text_alignment.jsonl", "text_model.json"]:
        source = args.work_dir / source_name
        destination = args.output_dir / source_name
        destination.write_bytes(source.read_bytes())

    typical_index = choose_typical_sample(base)
    write_typical_csv(args.output_dir / "typical_sample_alignment.csv", base, text_mask, text_alignment, typical_index)

    report = {
        "status": "pass",
        "sample_count": len(all_data["id"]),
        "unique_id_count": len(set(all_data["id"])),
        "shapes": {key: list(all_data[key].shape) for key in ["text", "audio", "vision", "time_bounds", "text_mask", "audio_mask", "vision_mask"]},
        "all_finite": {key: bool(np.isfinite(all_data[key]).all()) for key in ["text", "audio", "vision", "time_bounds"]},
        "mask_coverage": {
            key: float(all_data[key].mean()) for key in ["text_mask", "audio_mask", "vision_mask"]
        },
        "duration_s": {
            "min": float(all_data["durations"].min()),
            "median": float(np.median(all_data["durations"])),
            "max": float(all_data["durations"].max()),
        },
        "label_counts": dict(Counter(str(value) for value in all_data["annotations"])),
        "typical_sample_index": typical_index,
        "typical_sample_id": all_data["id"][typical_index],
        "output_file": output_pkl.name,
        "output_size_bytes": output_pkl.stat().st_size,
        "output_sha256": sha256_file(output_pkl),
    }
    (args.output_dir / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
