import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import torch

import train as baseline
import train_multiview as multiview


def evaluate_checkpoint(checkpoint_path, data_path, device, output_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    args = SimpleNamespace(**config)
    aligned = multiview.load_view(Path(data_path), "aligned")
    model = multiview.FusionModel(
        int(config.get("hidden", 128)),
        int(config.get("layers", 2)),
        float(config.get("dropout", 0.15)),
        bool(config.get("bidirectional", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    batch_size = int(config.get("batch_size", 64))
    amp_enabled = bool(config.get("amp", False)) and device.type == "cuda"
    test_data = aligned["splits"]["test"]
    results = {
        name: multiview.evaluate_fusion(model, test_data, batch_size, device, amp_enabled, scenario)
        for name, scenario in multiview.scenario_matrix().items()
    }
    payload = {
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "split": "test",
        "selection_was_done_on": "valid",
        "results": results,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint_path), "clean": results["clean"]}, indent=2))


def evaluate_baseline_checkpoint(checkpoint_path, data_path, device, output_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    with Path(data_path).open("rb") as handle:
        source = pickle.load(handle)
    test_data = baseline.prepare_split(source["test"], checkpoint["normalizers"])
    model = baseline.Baseline(
        int(config.get("hidden", 64)),
        float(config.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    batch_size = int(config.get("batch_size", 64))
    scores, _, _ = baseline.evaluate(model, test_data, batch_size, device)
    payload = {
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "split": "test",
        "selection_was_done_on": "valid",
        "clean": scores,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint_path), "clean": scores}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    for checkpoint in args.checkpoint:
        output_path = args.output / f"{checkpoint.parent.name}_test.json"
        evaluate_checkpoint(checkpoint, args.data, device, output_path)
    for checkpoint in args.baseline_checkpoint:
        output_path = args.output / f"{checkpoint.parent.name}_test.json"
        evaluate_baseline_checkpoint(checkpoint, args.data, device, output_path)


if __name__ == "__main__":
    main()
