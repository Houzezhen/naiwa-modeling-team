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


def lexical_spans(text: str, duration: float) -> list[dict]:
    matches = list(WORD_PATTERN.finditer(text))
    if not matches:
        return []
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
    cumulative = cumulative / cumulative[-1] * duration
    return [
        {
            "token": match.group(0),
            "char_start": match.start(),
            "char_end": match.end(),
            "start_s": float(cumulative[index]),
            "end_s": float(cumulative[index + 1]),
        }
        for index, match in enumerate(matches)
    ]


def token_time_span(offset_start: int, offset_end: int, lexical: list[dict]) -> tuple[float, float] | None:
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
    return float(best["start_s"]), float(best["end_s"])


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
        span = token_time_span(int(offset_start), int(offset_end), lexical)
        if span is None:
            continue
        start_s, end_s = span
        token_text = None
        for word in lexical:
            if word["start_s"] == start_s and word["end_s"] == end_s:
                token_text = word["token"]
                break
        first_bin = max(0, min(num_bins - 1, int(math.floor(start_s / duration * num_bins))))
        last_bin = max(0, min(num_bins - 1, int(math.ceil(end_s / duration * num_bins) - 1)))
        for bin_index in range(first_bin, last_bin + 1):
            overlap = max(0.0, min(end_s, edges[bin_index + 1]) - max(start_s, edges[bin_index]))
            if overlap <= 0:
                continue
            aligned[bin_index] += vector.astype(np.float32) * overlap
            weight_sum[bin_index] += overlap
            if token_text and token_text not in bin_tokens[bin_index]:
                bin_tokens[bin_index].append(token_text)
    valid = weight_sum > 0
    aligned[valid] /= weight_sum[valid, None].astype(np.float32)
    mask[valid] = 1
    return aligned, mask, bin_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DistilBERT features and align them to the common 50-bin timeline.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model", default="distilbert-base-uncased")
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModel.from_pretrained(args.model)
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
                lexical = lexical_spans(texts[sample_index], float(durations[sample_index]))
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
        "feature_dim": 768,
        "pooling": "mean of final four hidden layers, then overlap-weighted pooling into common time bins",
        "token_timing": "monotone transcript allocation weighted by token length and punctuation pause",
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
