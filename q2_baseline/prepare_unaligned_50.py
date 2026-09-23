import argparse
import pickle
from pathlib import Path

import numpy as np


MODALITIES = ("audio", "vision")


def aggregate_equal_duration(values, length, bins):
    length = min(max(int(length), 0), values.shape[0])
    result = np.zeros((bins, values.shape[1]), dtype=np.float32)
    mask = np.zeros(bins, dtype=bool)
    if length == 0:
        return result, mask
    source = np.nan_to_num(values[:length].astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    edges = np.linspace(0.0, float(length), bins + 1)
    for index in range(bins):
        start, end = edges[index], edges[index + 1]
        first = int(np.floor(start))
        last = min(length, int(np.ceil(end)))
        if last <= first:
            first = min(first, length - 1)
            last = first + 1
        frame_indices = np.arange(first, last)
        overlap = np.minimum(frame_indices + 1.0, end) - np.maximum(frame_indices, start)
        overlap = np.maximum(overlap, 0.0).astype(np.float32)
        total = float(overlap.sum())
        if total > 0:
            result[index] = (source[frame_indices] * overlap[:, None]).sum(axis=0) / total
            mask[index] = True
    return result, mask


def convert_split(split, bins):
    output = {
        key: value for key, value in split.items()
        if key not in {"audio", "vision", "audio_lengths", "vision_lengths"}
    }
    for modality in MODALITIES:
        values = np.asarray(split[modality])
        lengths = split[modality + "_lengths"]
        converted = np.zeros((len(values), bins, values.shape[-1]), dtype=np.float32)
        masks = np.zeros((len(values), bins), dtype=bool)
        for sample_index, length in enumerate(lengths):
            converted[sample_index], masks[sample_index] = aggregate_equal_duration(
                values[sample_index], length, bins
            )
        output[modality] = converted
        output[modality + "_mask"] = masks
        output[modality + "_lengths"] = list(lengths)
    output["alignment_rule"] = "length-aware equal-duration weighted bins"
    output["alignment_bins"] = bins
    return output


def main():
    parser = argparse.ArgumentParser(description="Convert padded variable-length AV features to a common timeline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("bins must be positive")
    with args.input.open("rb") as handle:
        source = pickle.load(handle)
    converted = {}
    for split_name, split in source.items():
        converted[split_name] = convert_split(split, args.bins)
        print(split_name, len(split["id"]), converted[split_name]["audio"].shape,
              converted[split_name]["vision"].shape, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(converted, handle, protocol=4)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
