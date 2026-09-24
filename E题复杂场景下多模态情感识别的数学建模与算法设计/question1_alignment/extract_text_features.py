from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", re.UNICODE)


def lexical_spans(text: str, duration: float, speech_segments: list[list[float]] | None = None) -> list[dict]:
    matches = list(WORD_PATTERN.finditer(text))
    if not matches or speech_segments == []:
        return []
    intervals = [[0.0, duration]] if speech_segments is None else speech_segments
    active_duration = sum(end - start for start, end in intervals)
    weights = []
    for match in matches:
        token = match.group(0)
        if token in {".", "!", "?"}:
            weight = 0.7
        elif token in {",", ";", ":"}:
            weight = 0.35
        else:
            alphanumeric = sum(character.isalnum() for character in token)
            weight = max(1.0, alphanumeric**0.70)
        weights.append(weight)
    cumulative = np.concatenate([[0.0], np.cumsum(weights, dtype=np.float64)])
    cumulative = cumulative / cumulative[-1] * active_duration
    spans = []
    elapsed = 0.0
    intervals_on_active_axis = []
    for start, end in intervals:
        intervals_on_active_axis.append((elapsed, elapsed + end - start, start))
        elapsed += end - start
    for index, match in enumerate(matches):
        pieces = []
        for active_start, active_end, real_start in intervals_on_active_axis:
            left = max(float(cumulative[index]), active_start)
            right = min(float(cumulative[index + 1]), active_end)
            if right > left:
                pieces.append([real_start + left - active_start, real_start + right - active_start])
        if not pieces:
            continue
        spans.append({
            "token": match.group(0),
            "char_start": match.start(),
            "char_end": match.end(),
            "start_s": pieces[0][0],
            "end_s": pieces[-1][1],
            "segments": pieces,
        })
    return spans


def token_time_span(offset_start: int, offset_end: int, lexical: list[dict]) -> dict | None:
    if offset_end <= offset_start:
        return None
    best = None
    best_overlap = 0
    for word in lexical:
        overlap = max(0, min(offset_end, word["char_end"]) - max(offset_start, word["char_start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best = word
    if best is None:
        return None
    return best


def align_hidden_states(
    token_vectors: np.ndarray,
    offsets: np.ndarray,
    lexical: list[dict],
    duration: float,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    feature_dim = token_vectors.shape[1]
    aligned = np.zeros((num_bins, feature_dim), dtype=np.float32)
    mask = np.zeros(num_bins, dtype=np.uint8)
    weight_sum = np.zeros(num_bins, dtype=np.float64)
    bin_tokens: list[list[str]] = [[] for _ in range(num_bins)]
    edges = np.linspace(0.0, duration, num_bins + 1)

    for vector, (offset_start, offset_end) in zip(token_vectors, offsets):
        word = token_time_span(int(offset_start), int(offset_end), lexical)
        if word is None:
            continue
        for start_s, end_s in word["segments"]:
            first_bin = max(0, min(num_bins - 1, int(math.floor(start_s / duration * num_bins))))
            last_bin = max(0, min(num_bins - 1, int(math.ceil(end_s / duration * num_bins) - 1)))
            for bin_index in range(first_bin, last_bin + 1):
                overlap = max(0.0, min(end_s, edges[bin_index + 1]) - max(start_s, edges[bin_index]))
                if overlap <= 0:
                    continue
                aligned[bin_index] += vector.astype(np.float32) * overlap
                weight_sum[bin_index] += overlap
                if word["token"] not in bin_tokens[bin_index]:
                    bin_tokens[bin_index].append(word["token"])
    valid = weight_sum > 0
    aligned[valid] /= weight_sum[valid, None].astype(np.float32)
    mask[valid] = 1
    return aligned, mask, bin_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DistilBERT features and align them to the common 50-bin timeline.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--revision", default="12040accade4e8a0f71eabdb258fecc2e7e948be")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    samples = json.loads((args.work_dir / "samples.json").read_text(encoding="utf-8"))
    texts = [sample["text"] for sample in samples]
    sample_ids = [sample["sample_id"] for sample in samples]
    durations = np.asarray([sample["duration_s"] for sample in samples], dtype=np.float64)
    num_bins = 50
    count = len(texts)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    model = AutoModel.from_pretrained(args.model, revision=args.revision)
    model.eval()
    model.to(device)

    text_features = np.zeros((count, num_bins, 768), dtype=np.float32)
    text_mask = np.zeros((count, num_bins), dtype=np.uint8)
    alignment_path = args.work_dir / "text_alignment.jsonl"
    with alignment_path.open("w", encoding="utf-8") as alignment_handle:
        for batch_start in range(0, count, args.batch_size):
            batch_end = min(count, batch_start + args.batch_size)
            batch_texts = texts[batch_start:batch_end]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping").cpu().numpy()
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model(**model_inputs, output_hidden_states=True, return_dict=True)
                hidden = torch.stack(output.hidden_states[-4:], dim=0).mean(dim=0).cpu().numpy()
            attention = encoded["attention_mask"].cpu().numpy()

            for local_index, sample_index in enumerate(range(batch_start, batch_end)):
                valid_tokens = attention[local_index].astype(bool)
                sample_vectors = hidden[local_index, valid_tokens]
                sample_offsets = offsets[local_index, valid_tokens]
                lexical = samples[sample_index].get("word_spans")
                if lexical is None:
                    lexical = lexical_spans(
                        texts[sample_index], float(durations[sample_index]), samples[sample_index].get("speech_segments")
                    )
                aligned, mask, bin_tokens = align_hidden_states(
                    sample_vectors,
                    sample_offsets,
                    lexical,
                    float(durations[sample_index]),
                    num_bins,
                )
                text_features[sample_index] = aligned
                text_mask[sample_index] = mask
                alignment_handle.write(
                    json.dumps(
                        {
                            "sample_index": sample_index,
                            "sample_id": sample_ids[sample_index],
                            "duration_s": float(durations[sample_index]),
                            "lexical_spans": lexical,
                            "bin_tokens": bin_tokens,
                            "valid_bins": int(mask.sum()),
                            "alignment_status": samples[sample_index].get("alignment_status", "full_duration_heuristic"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"Text [{batch_end:03d}/{count:03d}] device={device}")

    np.save(args.work_dir / "text_features.npy", text_features)
    np.save(args.work_dir / "text_mask.npy", text_mask)
    model_info = {
        "model": args.model,
        "model_revision": args.revision,
        "feature_dim": 768,
        "pooling": "mean of final four hidden layers, then overlap-weighted pooling into common time bins",
        "token_timing": "CTC word alignment intersected with energy activity; low-confidence and audio-conflict samples have no text timestamps" if any("word_spans" in sample for sample in samples) else "monotone allocation over energy-supported intervals if provided; not forced alignment",
        "ctc_model": "facebook/wav2vec2-base-960h" if any("word_spans" in sample for sample in samples) else None,
        "ctc_model_revision": next((sample["ctc_model_revision"] for sample in samples if "word_spans" in sample and "ctc_model_revision" in sample), None),
        "ctc_min_mean_label_posterior": 0.50 if any("word_spans" in sample for sample in samples) else None,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (args.work_dir / "text_model.json").write_text(json.dumps(model_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved text features with shape {text_features.shape}")


if __name__ == "__main__":
    main()
