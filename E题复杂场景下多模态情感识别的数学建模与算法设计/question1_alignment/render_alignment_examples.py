from __future__ import annotations

import argparse
import json
import pickle
import textwrap
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np


SAMPLE_IDS = [
    "-tPCytz4rww$_$18",
    "-iRBcNs9oI8$_$9",
    "-mJ2ud6oKI8$_$1",
]


def annotate(frame: av.VideoFrame, sample: dict, timestamp: float, frame_rate: Fraction) -> av.VideoFrame:
    bins = sample["bounds"]
    bin_index = min(49, max(0, int(timestamp / sample["duration"] * 50)))
    start, end = bins[bin_index]
    image = frame.to_ndarray(format="bgr24")
    height, width = image.shape[:2]
    output = np.full((height + 172, width, 3), (22, 24, 29), dtype=np.uint8)
    output[115 : 115 + height] = image
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.68, width / 950)
    small = min(0.58, width / 1100)
    green = (135, 231, 152)
    white = (240, 240, 240)
    warning = (90, 185, 255)
    words = " ".join(sample["tokens"][bin_index]) or "(no predicted word)"
    wrapped = textwrap.wrap(words, width=max(25, int(width / 13)))[:2]
    cv2.putText(output, f"{sample['id']}  |  t={timestamp:.2f}s  |  bin {bin_index + 1}/50", (12, 24), font, scale, white, 1, cv2.LINE_AA)
    cv2.putText(output, f"Predicted words: {wrapped[0] if wrapped else ''}", (12, 51), font, small, warning, 1, cv2.LINE_AA)
    if len(wrapped) > 1:
        cv2.putText(output, wrapped[1], (12, 75), font, small, warning, 1, cv2.LINE_AA)
    cv2.putText(output, f"Window [{start:.2f}, {end:.2f}) s  |  audio RMS {sample['rms'][bin_index]:.5f}", (12, 101), font, small, white, 1, cv2.LINE_AA)
    cv2.putText(output, f"Masks: text={sample['text_mask'][bin_index]} audio={sample['audio_mask'][bin_index]} vision={sample['vision_mask'][bin_index]}", (12, height + 139), font, small, green, 1, cv2.LINE_AA)
    cv2.putText(output, f"Status: {sample['status']} | CTC times need listening verification", (12, height + 162), font, small, warning, 1, cv2.LINE_AA)
    result = av.VideoFrame.from_ndarray(output, format="bgr24")
    result.pts = max(0, round(timestamp * float(frame_rate)))
    result.time_base = Fraction(1, frame_rate)
    return result


def render(source: Path, destination: Path, sample: dict) -> None:
    with av.open(str(source)) as reader, av.open(str(destination), mode="w") as writer:
        input_video = reader.streams.video[0]
        frame_rate = input_video.average_rate or Fraction(30, 1)
        video = writer.add_stream("libx264", rate=frame_rate, options={"preset": "veryfast", "crf": "27", "bf": "0"})
        video.width = input_video.codec_context.width
        video.height = input_video.codec_context.height + 172
        video.pix_fmt = "yuv420p"
        input_audio = reader.streams.audio[0] if reader.streams.audio else None
        output_audio = writer.add_stream_from_template(input_audio) if input_audio else None
        last_pts = -1
        frame_number = 0
        for packet in reader.demux([input_video, input_audio] if input_audio else [input_video]):
            if input_audio is not None and packet.stream.index == input_audio.index:
                if packet.dts is not None:
                    packet.stream = output_audio
                    writer.mux(packet)
                continue
            for frame in packet.decode():
                timestamp = float(frame.time) if frame.time is not None else frame_number / float(frame_rate)
                drawn = annotate(frame, sample, timestamp, frame_rate)
                drawn.pts = max(last_pts + 1, drawn.pts)
                last_pts = drawn.pts
                for encoded in video.encode(drawn):
                    writer.mux(encoded)
                frame_number += 1
        for encoded in video.encode():
            writer.mux(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render original-audio videos with question 1 alignment overlays.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alignment-output-dir", type=Path, default=Path(__file__).resolve().parent / "output_speech_aware")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    with (args.alignment_output_dir / "question1_aligned_50.pkl").open("rb") as handle:
        data = pickle.load(handle)["all"]
    alignment = {
        record["sample_id"]: record
        for record in map(json.loads, (args.alignment_output_dir / "text_alignment.jsonl").read_text(encoding="utf-8").splitlines())
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in SAMPLE_IDS:
        index = data["id"].index(sample_id)
        sample = {
            "id": sample_id,
            "duration": float(data["durations"][index]),
            "bounds": data["time_bounds"][index],
            "tokens": alignment[sample_id]["bin_tokens"],
            "rms": data["audio"][index, :, 72],
            "text_mask": data["text_mask"][index],
            "audio_mask": data["audio_mask"][index],
            "vision_mask": data["vision_mask"][index],
            "status": str(data["alignment_status"][index]) if "alignment_status" in data else "legacy_heuristic",
        }
        destination = args.output_dir / f"{sample_id.replace('$_$', '_clip_')}.mp4"
        render(args.data_root / data["source_file"][index], destination, sample)
        print(destination)


if __name__ == "__main__":
    main()
