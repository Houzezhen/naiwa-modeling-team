from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import av
import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Export representative source frames for the typical alignment sample.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads((args.output_dir / "quality_report.json").read_text(encoding="utf-8"))
    sample_id = report["typical_sample_id"]
    with (args.output_dir / "question1_aligned_50.pkl").open("rb") as handle:
        data = pickle.load(handle)["all"]
    sample_index = data["id"].index(sample_id)
    source_path = args.data_root / data["source_file"][sample_index]
    bin_indices = [0, 12, 25, 37, 49]
    targets = [float(data["vision_timestamps"][sample_index, index]) for index in bin_indices]
    frames: list[np.ndarray | None] = [None] * len(targets)
    distances = [float("inf")] * len(targets)

    with av.open(str(source_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        for frame_index, frame in enumerate(container.decode(video=0)):
            timestamp = float(frame.time) if frame.time is not None else frame_index / fps
            for target_index, target in enumerate(targets):
                distance = abs(timestamp - target)
                if distance < distances[target_index]:
                    frames[target_index] = frame.to_ndarray(format="bgr24")
                    distances[target_index] = distance

    frame_dir = args.output_dir / "typical_sample_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for bin_index, target, frame, distance in zip(bin_indices, targets, frames, distances):
        if frame is None:
            raise RuntimeError(f"Could not decode frame for bin {bin_index}")
        filename = f"bin_{bin_index:02d}_time_{target:.3f}s.jpg"
        destination = frame_dir / filename
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not encoded:
            raise RuntimeError(f"Could not write {destination}")
        destination.write_bytes(buffer.tobytes())
        bounds = data["time_bounds"][sample_index, bin_index]
        rows.append(
            {
                "sample_id": sample_id,
                "bin_index": bin_index,
                "bin_start_s": f"{float(bounds[0]):.6f}",
                "bin_end_s": f"{float(bounds[1]):.6f}",
                "frame_timestamp_s": f"{target:.6f}",
                "nearest_frame_error_s": f"{distance:.6f}",
                "frame_file": str(Path("typical_sample_frames") / filename),
            }
        )

    with (args.output_dir / "typical_sample_frames.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} frames for {sample_id}")


if __name__ == "__main__":
    main()
