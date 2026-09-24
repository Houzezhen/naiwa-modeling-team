from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from extract_av_features import audio_bin_features, decode_audio


def activity_segments(audio: np.ndarray, valid_duration: float, sample_rate: int) -> tuple[list[list[float]], float]:
    signal = audio[: int(round(valid_duration * sample_rate))]
    frame_size = int(0.02 * sample_rate)
    if len(signal) < frame_size or not np.any(signal):
        return [], 0.0
    energy = np.array(
        [np.sqrt(np.mean(signal[start : start + frame_size].astype(np.float64) ** 2))
         for start in range(0, len(signal) - frame_size + 1, frame_size)]
    )
    threshold = max(0.003, 0.12 * float(np.percentile(energy, 90)))
    active = energy >= threshold
    for index in range(1, len(active) - 2):
        if active[index - 1] and not active[index] and active[index + 1 : index + 3].any():
            active[index] = True
    segments = []
    for index in range(len(active)):
        if not active[index] or (index and active[index - 1]):
            continue
        end = index + 1
        while end < len(active) and active[end]:
            end += 1
        start_s = max(0.0, index * 0.02 - 0.02)
        end_s = min(valid_duration, end * 0.02 + 0.02)
        if end_s - start_s >= 0.08:
            if segments and start_s - segments[-1][1] <= 0.08:
                segments[-1][1] = end_s
            else:
                segments.append([start_s, end_s])
    return segments, threshold


def window_coverage(segments: list[list[float]], time_bounds: np.ndarray) -> np.ndarray:
    coverage = np.zeros(len(time_bounds), dtype=np.float32)
    for index, (start, end) in enumerate(time_bounds):
        overlap = sum(max(0.0, min(end, right) - max(start, left)) for left, right in segments)
        coverage[index] = min(1.0, overlap / (end - start))
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare conservative audio-evidence-guided text alignment.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-work-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with (args.source_work_dir / "base_features.pkl").open("rb") as handle:
        base = pickle.load(handle)
    samples = json.loads((args.source_work_dir / "samples.json").read_text(encoding="utf-8"))
    count, bins = base["audio_mask"].shape
    waveforms_dir = args.work_dir / "audio_waveforms"
    waveforms_dir.mkdir(exist_ok=True)
    if len(samples) != count:
        raise ValueError("Source samples and features do not match")
    coverage = np.zeros((count, bins), dtype=np.float32)
    speech_coverage = np.zeros((count, bins), dtype=np.float32)
    activity_records = []
    for index, sample in enumerate(samples):
        source = args.data_root / base["source_file"][index]
        audio, valid_duration = decode_audio(source, 16000, float(base["durations"][index]))
        np.save(waveforms_dir / f"{index:03d}.npy", audio[:int(round(valid_duration * 16000))])
        if abs(valid_duration - float(base["audio_valid_durations"][index])) > 0.002:
            raise ValueError(f"Changed decoded duration for {sample['sample_id']}")
        segments, threshold = activity_segments(audio, valid_duration, 16000)
        bounds = base["time_bounds"][index]
        speech_coverage[index] = window_coverage(segments, bounds)
        samples[index]["speech_segments"] = segments
        samples[index]["alignment_status"] = (
            "transcript_audio_conflict" if not np.any(audio[: int(valid_duration * 16000)]) and sample["text"].strip()
            else "no_energy_support" if not segments and sample["text"].strip()
            else "energy_supported_unverified" if segments else "no_transcript"
        )
        for bin_index, (start, end) in enumerate(bounds):
            start_sample = int(round(float(start) * 16000))
            end_sample = min(int(round(float(end) * 16000)), int(round(valid_duration * 16000)))
            actual_samples = max(0, end_sample - start_sample)
            coverage[index, bin_index] = min(1.0, actual_samples / ((end - start) * 16000))
            base["audio"][index, bin_index] = 0
            base["audio_mask"][index, bin_index] = 0
            if actual_samples >= 400:
                base["audio"][index, bin_index] = audio_bin_features(audio[start_sample:end_sample], 16000)
                base["audio_mask"][index, bin_index] = 1
        activity_records.append({
            "sample_index": index,
            "sample_id": sample["sample_id"],
            "status": samples[index]["alignment_status"],
            "audio_valid_duration_s": valid_duration,
            "energy_rms_threshold": threshold,
            "candidate_activity_segments_s": segments,
            "candidate_active_bins": int((speech_coverage[index] > 0).sum()),
            "text_preserved": sample["text"],
        })
        print(f"[{index + 1:03d}/{count:03d}] {sample['sample_id']} {samples[index]['alignment_status']}")
    base["audio_coverage"] = coverage
    base["speech_coverage"] = speech_coverage
    base["speech_mask"] = (speech_coverage > 0).astype(np.uint8)
    with (args.work_dir / "base_features.pkl").open("wb") as handle:
        pickle.dump(base, handle, protocol=4)
    (args.work_dir / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.work_dir / "speech_activity.jsonl").open("w", encoding="utf-8") as handle:
        for record in activity_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.source_work_dir / "feature_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        manifest = list(reader)
    index_by_id = {sample_id: index for index, sample_id in enumerate(base["id"])}
    for row in manifest:
        index = index_by_id[row["sample_id"]]
        if row["modality"] == "audio":
            row["valid_bins"] = str(int(base["audio_mask"][index].sum()))
            row["padding_rule"] = "zero when audio_mask=0; partial coverage measured separately"
        elif row["modality"] == "vision":
            row["valid_duration_s"] = f"{float(base['vision_mask'][index].sum()) * float(base['durations'][index]) / bins:.6f}"
    with (args.work_dir / "feature_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    (args.work_dir / "processing_log.jsonl").write_bytes((args.source_work_dir / "processing_log.jsonl").read_bytes())
    print("Alignment status counts:", dict(Counter(row["status"] for row in activity_records)))


if __name__ == "__main__":
    main()
