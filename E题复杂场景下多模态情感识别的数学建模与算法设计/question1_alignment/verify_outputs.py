from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify question 1 aligned output files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_pkl = args.output_dir / "question1_aligned_50.pkl"
    with output_pkl.open("rb") as handle:
        payload = pickle.load(handle)
    require(set(payload) == {"all", "metadata"}, "Unexpected top-level schema")
    data = payload["all"]
    require(len(data["id"]) == 100, "Expected 100 samples")
    require(len(set(data["id"])) == 100, "Sample IDs are not unique")
    require(data["text"].shape == (100, 50, 768), "Bad text shape")
    require(data["audio"].shape == (100, 50, 74), "Bad audio shape")
    require(data["vision"].shape == (100, 50, 35), "Bad vision shape")
    require(data["time_bounds"].shape == (100, 50, 2), "Bad time-bound shape")
    for key in ["text", "audio", "vision", "time_bounds"]:
        require(bool(np.isfinite(data[key]).all()), f"Non-finite values in {key}")
    for modality in ["text", "audio", "vision"]:
        mask = data[f"{modality}_mask"].astype(bool)
        features = data[modality]
        require(bool(np.all(features[~mask] == 0)), f"Nonzero padding in {modality}")
        require(bool(np.all(np.linalg.norm(features[mask], axis=-1) > 0)), f"Zero valid rows in {modality}")
    starts = data["time_bounds"][:, :, 0]
    ends = data["time_bounds"][:, :, 1]
    require(bool(np.all(ends > starts)), "Non-positive time windows")
    require(bool(np.allclose(ends[:, :-1], starts[:, 1:], atol=1e-6)), "Time windows are not contiguous")
    require(bool(np.allclose(starts[:, 0], 0.0, atol=1e-7)), "Timelines do not start at zero")
    require(bool(np.allclose(ends[:, -1], data["durations"], atol=1e-5)), "Timelines do not end at media duration")

    with (args.output_dir / "feature_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    require(len(manifest_rows) == 300, "Manifest must contain 100 samples x 3 modalities")
    require(len({(row["sample_id"], row["modality"]) for row in manifest_rows}) == 300, "Manifest keys are not unique")

    with (args.output_dir / "typical_sample_frames.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        frame_rows = list(csv.DictReader(handle))
    require(len(frame_rows) == 5, "Expected five typical-sample frame records")
    for row in frame_rows:
        require((args.output_dir / row["frame_file"]).is_file(), f"Missing typical frame: {row['frame_file']}")

    report = json.loads((args.output_dir / "quality_report.json").read_text(encoding="utf-8"))
    require(report["status"] == "pass", "Quality report status is not pass")
    print("PASS: 100 samples, exact tensor shapes, finite features, valid masks, contiguous time bounds, 300 manifest rows, and 5 typical frames.")


if __name__ == "__main__":
    main()
