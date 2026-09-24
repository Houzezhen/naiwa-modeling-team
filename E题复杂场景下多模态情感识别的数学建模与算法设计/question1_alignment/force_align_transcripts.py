from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def transcript_tokens(text: str, vocabulary: dict[str, int]) -> tuple[list[int], list[dict]]:
    labels = []
    words = []
    for match in WORD_PATTERN.finditer(text):
        spelling = re.sub("[^A-Z]", "", match.group().upper())
        if not spelling or any(letter not in vocabulary for letter in spelling):
            continue
        if labels:
            labels.append(vocabulary["|"])
        start = len(labels)
        labels.extend(vocabulary[letter] for letter in spelling)
        words.append({"token": match.group(), "char_start": match.start(), "char_end": match.end(), "first_label": start, "last_label": len(labels) - 1})
    return labels, words


def viterbi_ctc(logits: np.ndarray, labels: list[int], blank_id: int) -> tuple[list[tuple[int, int, float]], float] | None:
    frames, vocabulary = logits.shape
    count = len(labels)
    if count == 0 or frames < count:
        return None
    logit_max = logits.max(axis=1, keepdims=True)
    log_probs = logits - logit_max - np.log(np.exp(logits - logit_max).sum(axis=1, keepdims=True))
    sequence = np.full(count * 2 + 1, blank_id, dtype=np.int32)
    sequence[1::2] = labels
    states = len(sequence)
    path = np.full((frames, states), -1, dtype=np.int8)
    scores = np.full(states, -np.inf, dtype=np.float64)
    scores[0] = log_probs[0, blank_id]
    scores[1] = log_probs[0, labels[0]]
    skip = np.zeros(states, dtype=bool)
    skip[3::2] = sequence[3::2] != sequence[1:-2:2]
    for time in range(1, frames):
        stay = scores
        previous = np.r_[-np.inf, scores[:-1]]
        jump = np.r_[-np.inf, -np.inf, scores[:-2]]
        jump[~skip] = -np.inf
        options = np.stack((stay, previous, jump))
        decisions = np.argmax(options, axis=0)
        scores = options[decisions, np.arange(states)] + log_probs[time, sequence]
        path[time] = decisions
    state = states - 1 if scores[-1] >= scores[-2] else states - 2
    if not np.isfinite(scores[state]):
        return None
    positions = [[] for _ in labels]
    for time in range(frames - 1, -1, -1):
        if state % 2:
            positions[state // 2].append(time)
        state -= int(path[time, state]) if time else 0
    if any(not position for position in positions):
        return None
    spans = [(min(position), max(position) + 1, float(np.exp(log_probs[position, label]).mean())) for position, label in zip(positions, labels)]
    return spans, float(np.mean([item[2] for item in spans]))


def main() -> None:
    parser = argparse.ArgumentParser(description="CTC-forced alignment of supplied transcripts against the supplied audio.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--revision", default="22aad52d435eb6dbaf354bdad9b0da84ce7d6156")
    parser.add_argument("--min-confidence", type=float, default=0.50)
    args = parser.parse_args()
    samples_path = args.work_dir / "samples.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    processor = Wav2Vec2Processor.from_pretrained(args.model, revision=args.revision)
    model = Wav2Vec2ForCTC.from_pretrained(args.model, revision=args.revision).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    vocab = processor.tokenizer.get_vocab()
    if "|" not in vocab:
        raise ValueError("The CTC model lacks a word separator")
    for index, sample in enumerate(samples):
        if "ctc_mean_label_posterior" in sample:
            sample["ctc_model_revision"] = args.revision
        if sample["alignment_status"] == "ctc_aligned_unverified":
            if sample["ctc_mean_label_posterior"] < args.min_confidence:
                sample["alignment_status"] = "ctc_low_confidence"
                sample["speech_segments"] = []
                del sample["word_spans"]
            else:
                supported_words = []
                for word in sample["word_spans"]:
                    segments = [
                        [max(word["start_s"], left), min(word["end_s"], right)]
                        for left, right in sample["speech_segments"]
                        if min(word["end_s"], right) > max(word["start_s"], left)
                    ]
                    if segments:
                        word["segments"] = segments
                        word["start_s"], word["end_s"] = segments[0][0], segments[-1][1]
                        supported_words.append(word)
                sample["word_spans"] = supported_words
                if not supported_words:
                    sample["alignment_status"] = "ctc_no_energy_supported_word"
                    sample["speech_segments"] = []
        if sample["alignment_status"] != "energy_supported_unverified":
            continue
        labels, words = transcript_tokens(sample["text"], vocab)
        if not words:
            sample["alignment_status"] = "ctc_unsupported_transcript"
            sample["speech_segments"] = []
            continue
        waveform = np.load(args.work_dir / "audio_waveforms" / f"{index:03d}.npy")
        valid_duration = len(waveform) / 16000
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            emissions = model(inputs.input_values.to(device)).logits[0].cpu().numpy()
        result = viterbi_ctc(emissions, labels, model.config.pad_token_id)
        if result is None:
            sample["alignment_status"] = "ctc_failed"
            sample["speech_segments"] = []
            continue
        spans, confidence = result
        sample["ctc_mean_label_posterior"] = confidence
        sample["ctc_model"] = args.model
        sample["ctc_model_revision"] = args.revision
        if confidence < args.min_confidence:
            sample["alignment_status"] = "ctc_low_confidence"
            sample["speech_segments"] = []
            print(f"[{index + 1:03d}/{len(samples):03d}] {sample['sample_id']} LOW {confidence:.3f}", flush=True)
            continue
        word_spans = []
        for word in words:
            aligned = spans[word["first_label"] : word["last_label"] + 1]
            start = aligned[0][0] / len(emissions) * valid_duration
            end = aligned[-1][1] / len(emissions) * valid_duration
            supported = [
                [max(start, left), min(end, right)]
                for left, right in sample["speech_segments"]
                if min(end, right) > max(start, left)
            ]
            if not supported:
                continue
            word_spans.append({"token": word["token"], "char_start": word["char_start"], "char_end": word["char_end"],
                               "start_s": supported[0][0], "end_s": supported[-1][1], "segments": supported,
                               "ctc_confidence": float(np.mean([item[2] for item in aligned]))})
        if not word_spans:
            sample["alignment_status"] = "ctc_no_energy_supported_word"
            sample["speech_segments"] = []
            continue
        sample["word_spans"] = word_spans
        sample["alignment_status"] = "ctc_aligned_unverified"
        print(f"[{index + 1:03d}/{len(samples):03d}] {sample['sample_id']} CTC {confidence:.3f}", flush=True)
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
