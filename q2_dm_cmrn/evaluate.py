"""评估入口：在验证集/测试划分上按多个缺失场景评估检查点。

示例：
    python evaluate.py --checkpoint outputs/dm_cmrn/best.pt --split test
    python evaluate.py --checkpoint outputs/cross_attn/best.pt --split valid --output outputs/eval_cross_attn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.data import Scenario  # noqa: E402
from src.engine import (load_checkpoint, load_prepared_splits, make_loader, resolve_device,  # noqa: E402
                        run_epoch, set_seed)
from src.paths import DATA_KINDS, MODALITIES  # noqa: E402


def default_scenarios() -> dict:
    """默认场景集：干净 + 单模态缺失 + 双模态缺失 + 三模态缺失（缺失率 30%）。"""
    scenarios = {"clean": Scenario(mode="none")}
    for name in MODALITIES:
        scenarios[f"single_{name}_30"] = Scenario(mode=name, position="random", ratio=0.3)
    scenarios["double_text+audio_30"] = Scenario(mode="text+audio", ratio=0.3)
    scenarios["double_text+vision_30"] = Scenario(mode="text+vision", ratio=0.3)
    scenarios["double_audio+vision_30"] = Scenario(mode="audio+vision", ratio=0.3)
    scenarios["triple_30"] = Scenario(mode="text+audio+vision", ratio=0.3)
    return scenarios


def evaluate_scenarios(model, prepared: dict, device, batch_size: int, scenarios: dict,
                       seed: int = 42, with_clean: bool = False) -> dict:
    results = {}
    settings = {"reg_weight": 0.5, "rec_weight": 0.0, "distill_weight": 0.0, "class_weights": None,
                "rec_target": "context", "ctx_span": 6, "similarity_weight": 0.0, "clip": 1.0}
    for name, scenario in scenarios.items():
        _, loader = make_loader(prepared, batch_size, seed, train_mode=False, scenario=scenario,
                               with_clean=with_clean)
        results[name] = run_epoch(model, loader, None, device, settings, train=False)
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--data-kind", choices=tuple(DATA_KINDS), default="aligned")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    data_path = args.data or DATA_KINDS[args.data_kind]
    print(f"检查点：{args.checkpoint}\n模型：{payload['model_name']} {payload.get('model_kwargs')}\n"
          f"数据：{data_path}（{args.split}）", flush=True)
    bundle = load_prepared_splits(Path(data_path), need_test=args.split == "test")
    prepared = bundle["prepared"]
    results = evaluate_scenarios(model, prepared[args.split], device, args.batch_size,
                                default_scenarios(), args.seed)
    rows = [{"scenario": name, **{key: value for key, value in metrics.items()
                                  if key != "prediction_distribution"}}
            for name, metrics in results.items()]
    frame = pd.DataFrame(rows)
    print(frame[["scenario", "accuracy", "macro_f1", "mae", "pearson", "ccc"]].to_string(index=False),
          flush=True)
    output = args.output or (args.checkpoint.parent / f"eval_{args.split}")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"metrics_{args.split}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    frame.to_csv(output / f"metrics_{args.split}.csv", index=False, encoding="utf-8-sig")
    print(f"已写入：{output}", flush=True)


if __name__ == "__main__":
    main()
