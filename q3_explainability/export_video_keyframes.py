from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import av


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_nearest(video_path: Path, target_s: float, output_path: Path) -> tuple[float, float, float]:
    container = av.open(str(video_path))
    best_image = None
    best_timestamp = None
    duration = float(container.duration / av.time_base) if container.duration else 0.0
    for frame in container.decode(video=0):
        timestamp = float(frame.time) if frame.time is not None else 0.0
        if best_timestamp is None or abs(timestamp - target_s) < abs(best_timestamp - target_s):
            best_timestamp = timestamp
            best_image = frame.to_image()
    container.close()
    if best_image is None or best_timestamp is None:
        raise RuntimeError(f"No decodable video frame: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_image.save(output_path, quality=92)
    return duration, best_timestamp, abs(best_timestamp - target_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export actual video keyframes for Q3 visual evidence.")
    parser.add_argument("--explanations", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.explanations.open("r", encoding="utf-8-sig", newline="")))
    output_dir = args.output / "video_keyframes"
    manifest = []
    for row in rows:
        if row.get("modality") != "vision":
            continue
        sample_id = row["id"]
        video_path = args.video_dir / f"{sample_id}.mp4"
        target_s = (float(row["start_time"]) + float(row["end_time"])) / 2.0
        frame_path = output_dir / f"{sample_id}_rank_{int(row['rank']):02d}_window_{int(row['window_index']):02d}.jpg"
        duration, actual_s, error_s = extract_nearest(video_path, target_s, frame_path)
        manifest.append({
            "id": sample_id,
            "rank": row["rank"],
            "window_index": row["window_index"],
            "source_video": str(video_path),
            "video_duration_s": f"{duration:.6f}",
            "target_time_s": f"{target_s:.6f}",
            "actual_frame_time_s": f"{actual_s:.6f}",
            "nearest_frame_error_s": f"{error_s:.6f}",
            "frame_path": str(frame_path),
            "frame_sha256": sha256_file(frame_path),
        })
    manifest_path = args.output / "video_keyframe_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Exported {len(manifest)} video keyframes to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
