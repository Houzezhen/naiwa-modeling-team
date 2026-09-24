from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute reproducible alignment quality metrics.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    with (args.output_dir / "question1_aligned_50.pkl").open("rb") as handle:
        data = pickle.load(handle)["all"]
    alignment_rows = [
        json.loads(line)
        for line in (args.output_dir / "text_alignment.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    sample_count = len(data["id"])
    window_count = int(data["text_mask"].shape[1])
    time_bounds = np.asarray(data["time_bounds"], dtype=np.float64)
    durations = np.asarray(data["durations"], dtype=np.float64)
    starts = time_bounds[:, :, 0]
    ends = time_bounds[:, :, 1]
    status_counts = Counter(str(status) for status in data["alignment_status"])

    temporal_checks = {
        "starts_at_zero_rate": float(np.mean(np.isclose(starts[:, 0], 0.0, atol=1e-7))),
        "contiguous_windows_rate": float(np.mean(np.all(np.isclose(ends[:, :-1], starts[:, 1:], atol=1e-6), axis=1))),
        "ends_at_media_duration_rate": float(np.mean(np.isclose(ends[:, -1], durations, atol=1e-5))),
        "positive_window_rate": float(np.mean(np.all(ends > starts, axis=1))),
    }
    mask_coverage = {
        "text": float(np.asarray(data["text_mask"], dtype=np.float64).mean()),
        "audio": float(np.asarray(data["audio_mask"], dtype=np.float64).mean()),
        "vision": float(np.asarray(data["vision_mask"], dtype=np.float64).mean()),
        "speech_candidate": float(np.asarray(data["speech_mask"], dtype=np.float64).mean()),
    }

    id_to_index = {sample_id: index for index, sample_id in enumerate(data["id"])}
    span_range_ok: list[bool] = []
    span_positive_duration: list[bool] = []
    sample_monotonic: list[bool] = []
    sample_nonoverlap: list[bool] = []
    valid_bin_matches: list[bool] = []
    token_confidences: list[float] = []
    spans_by_status = Counter()
    text_mask = np.asarray(data["text_mask"])

    for row in alignment_rows:
        duration = float(row["duration_s"])
        spans = row.get("lexical_spans", [])
        span_starts = [float(span["start_s"]) for span in spans]
        span_ends = [float(span["end_s"]) for span in spans]
        sample_monotonic.append(all(left <= right + 1e-9 for left, right in zip(span_starts, span_starts[1:])))
        sample_nonoverlap.append(all(left >= right - 1e-9 for left, right in zip(span_starts[1:], span_ends)))
        for span in spans:
            start_s = float(span["start_s"])
            end_s = float(span["end_s"])
            span_range_ok.append(-1e-6 <= start_s <= duration + 1e-6 and -1e-6 <= end_s <= duration + 1e-6)
            span_positive_duration.append(end_s > start_s)
            if "ctc_confidence" in span:
                token_confidences.append(float(span["ctc_confidence"]))
        status = str(row["alignment_status"])
        spans_by_status[status] += len(spans)
        sample_index = id_to_index[row["sample_id"]]
        valid_bin_matches.append(int(row.get("valid_bins", 0)) == int(text_mask[sample_index].sum()))

    metrics = {
        "schema_version": "question1-alignment-metrics-v1",
        "sample_count": sample_count,
        "window_count_per_sample": window_count,
        "window_total": sample_count * window_count,
        "mask_coverage": mask_coverage,
        "alignment_status_counts": dict(status_counts),
        "alignment_status_rates": {key: float(value / sample_count) for key, value in status_counts.items()},
        "temporal_invariants_sample_rate": temporal_checks,
        "text_alignment_records": len(alignment_rows),
        "text_alignment_valid_bin_match_rate": float(np.mean(valid_bin_matches)) if valid_bin_matches else None,
        "lexical_span_count": len(span_range_ok),
        "lexical_span_boundary_in_range_rate": float(np.mean(span_range_ok)) if span_range_ok else None,
        "lexical_span_positive_duration_rate": float(np.mean(span_positive_duration)) if span_positive_duration else None,
        "sample_monotonic_span_order_rate": float(np.mean(sample_monotonic)) if sample_monotonic else None,
        "sample_nonoverlapping_span_rate": float(np.mean(sample_nonoverlap)) if sample_nonoverlap else None,
        "token_ctc_confidence": summarize(token_confidences),
        "lexical_spans_by_status": dict(spans_by_status),
        "gold_standard_metrics": {
            "word_onset_mae_ms": None,
            "word_offset_mae_ms": None,
            "boundary_within_10ms_rate": None,
            "boundary_within_20ms_rate": None,
            "boundary_within_50ms_rate": None,
            "word_interval_iou": None,
            "reason": "No manually annotated word-level start/end timestamps are available.",
        },
    }
    text = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if args.write:
        (args.output_dir / "alignment_metrics.json").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
