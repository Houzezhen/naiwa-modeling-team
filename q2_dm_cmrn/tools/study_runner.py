"""多种子复算与超参筛选：训练（子进程）→ 同口径评估 → 均值±标准差汇总。

用法：
    # ① 多种子复算：dm_cmrn 与 cross_attn 各跑 5 个种子（每个种子独立训练教师）
    python -u tools/study_runner.py --study seeds --seeds 42 43 44 45 46

    # ② 超参筛选：rec_weight / ctx_span，并放宽早停（min_delta=0, patience=6）
    python -u tools/study_runner.py --study tune --seeds 42

    # 断点续跑：已存在 best.pt 的组合直接复用
    python -u tools/study_runner.py --study seeds --seeds 42 43 44 45 46 --reuse

产出（<root> = outputs/_study/<study>）：
    <root>/records.csv            每次运行的配置与训练信息
    <root>/metrics.csv            逐 (配置, 种子, 场景) 的指标
    <root>/summary.md             均值 ± 标准差汇总（论文用）
    <root>/summary.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from evaluate import default_scenarios, evaluate_scenarios  # noqa: E402
from src.engine import load_checkpoint, load_prepared_splits, resolve_device, set_seed  # noqa: E402
from src.paths import DATA_KINDS, OUTPUT_ROOT  # noqa: E402

# 汇总时使用的场景（干净 + 三模态 30% 缺失 + 三个单模态 30% 缺失）
SCENARIOS = ("clean", "triple_30", "single_text_30", "single_audio_30", "single_vision_30")

# ① 多种子复算的配置
SEED_CONFIGS = {
    "cross_attn": {"model": "cross_attn", "args": []},
    "dm_cmrn": {"model": "dm_cmrn", "args": []},
}

# ② 超参筛选的配置（都以 seed 给定；配合 --min-delta 0 --patience 6 让模型训满）
TUNE_CONFIGS = {
    "rec0.3_span6_long": {"model": "dm_cmrn", "args": []},                        # 对照组：同配置但放宽早停
    "rec0.05_span6": {"model": "dm_cmrn", "args": ["--rec-weight", "0.05"]},
    "rec0.1_span6": {"model": "dm_cmrn", "args": ["--rec-weight", "0.1"]},
    "rec0.3_span3": {"model": "dm_cmrn", "args": ["--ctx-span", "3"]},
    "rec0.3_span12": {"model": "dm_cmrn", "args": ["--ctx-span", "12"]},
}

STUDIES = {"seeds": SEED_CONFIGS, "tune": TUNE_CONFIGS}


def train_command(model: str, run_dir: Path, teacher: Path | None, args, extra: list[str]) -> list[str]:
    command = [sys.executable, "-u", str(HERE / "train.py"), "--model", model,
               "--epochs", str(args.epochs), "--patience", str(args.patience),
               "--min-delta", str(args.min_delta), "--hidden", str(args.hidden),
               "--layers", str(args.layers), "--dropout", str(args.dropout),
               "--batch-size", str(args.batch_size), "--seed", str(args.seed),
               "--device", args.device, "--output", str(run_dir)]
    if model == "dm_cmrn":
        command += ["--stage", "student"]
        if teacher is not None:
            command += ["--teacher", str(teacher), "--distill-weight", "1.0"]
    command += extra
    return command


def teacher_command(run_dir: Path, args) -> list[str]:
    return [sys.executable, "-u", str(HERE / "train.py"), "--model", "cross_attn", "--stage", "teacher",
            "--clean-only", "--epochs", str(args.teacher_epochs), "--patience", "3", "--min-delta", "0",
            "--hidden", str(args.hidden), "--layers", str(args.layers), "--dropout", str(args.dropout),
            "--batch-size", str(args.batch_size), "--seed", str(args.seed), "--device", args.device,
            "--output", str(run_dir)]


def run_training(command: list[str], log_path: Path) -> tuple[bool, float]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=str(HERE), stdout=handle, stderr=subprocess.STDOUT)
    return completed.returncode == 0, round(time.time() - started, 1)


def train_all(args, root: Path) -> list[dict]:
    """按 (配置 × 种子) 训练；dm_cmrn 的教师按种子独立训练。"""
    configs = STUDIES[args.study]
    records = []
    needs_teacher = any(spec["model"] == "dm_cmrn" for spec in configs.values())
    for seed in args.seeds:
        args.seed = seed
        teacher_dir = root / f"teacher_seed{seed}"
        teacher_path = teacher_dir / "best.pt"
        teacher_seconds = 0.0
        if needs_teacher and not (teacher_path.is_file() and args.reuse):
            print(f"[seed {seed}] 训练教师 → {teacher_dir.name}", flush=True)
            ok, teacher_seconds = run_training(teacher_command(teacher_dir, args),
                                               root / "logs" / f"teacher_seed{seed}.log")
            if not ok or not teacher_path.is_file():
                print(f"[seed {seed}] 教师训练失败，跳过该种子", flush=True)
                continue
        for name, spec in configs.items():
            run_dir = root / f"{name}_seed{seed}"
            record = {"config": name, "model": spec["model"], "seed": seed,
                      "extra": " ".join(spec["args"]), "run_dir": str(run_dir),
                      "teacher": str(teacher_path) if spec["model"] == "dm_cmrn" else "",
                      "teacher_seconds": teacher_seconds}
            if (run_dir / "best.pt").is_file() and args.reuse:
                print(f"[seed {seed}] {name}: 复用已有权重", flush=True)
                record.update({"trained": False, "seconds": 0.0})
            else:
                command = train_command(spec["model"], run_dir, teacher_path, args, spec["args"])
                print(f"[seed {seed}] {name}: 开始训练（extra='{record['extra']}'）", flush=True)
                ok, seconds = run_training(command, root / "logs" / f"{name}_seed{seed}.log")
                if not ok or not (run_dir / "best.pt").is_file():
                    print(f"[seed {seed}] {name}: 训练失败（见 logs）", flush=True)
                    continue
                record.update({"trained": True, "seconds": seconds})
            records.append(record)
    return records


def evaluate_all(args, root: Path, records: list[dict]) -> pd.DataFrame:
    """一次性加载数据，逐个检查点在 test 划分的同一套场景下评估。"""
    set_seed(args.seeds[0])
    device = resolve_device(args.device)
    data_path = args.data or DATA_KINDS[args.data_kind]
    bundle = load_prepared_splits(Path(data_path), need_test=True)
    prepared = bundle["prepared"]["test"]
    all_scenarios = default_scenarios()
    scenarios = {name: all_scenarios[name] for name in SCENARIOS}
    rows = []
    for record in records:
        checkpoint = Path(record["run_dir"]) / "best.pt"
        if not checkpoint.is_file():
            continue
        model, payload = load_checkpoint(checkpoint, device)
        results = evaluate_scenarios(model, prepared, device, args.batch_size, scenarios, record["seed"])
        for scenario, metrics in results.items():
            rows.append({"config": record["config"], "seed": record["seed"], "scenario": scenario,
                         "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                         "weighted_f1": metrics["weighted_f1"], "mae": metrics["mae"],
                         "pearson": metrics["pearson"], "ccc": metrics["ccc"],
                         "best_epoch": payload.get("metrics", {}).get("epoch", ""),
                         "train_seconds": record.get("seconds", "")})
        print(f"评估 {record['config']} seed={record['seed']} "
              f"clean F1={results['clean']['macro_f1']:.4f} "
              f"triple_30 F1={results['triple_30']['macro_f1']:.4f} "
              f"(best epoch {payload.get('metrics', {}).get('epoch', '')})", flush=True)
        del model
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, path: Path) -> dict:
    metrics = ["accuracy", "macro_f1", "mae", "pearson", "ccc"]
    summary = {}
    for config, block in frame.groupby("config"):
        summary[config] = {"seeds": int(block["seed"].nunique()),
                           "best_epoch_mean": float(pd.to_numeric(block["best_epoch"],
                                                                  errors="coerce").mean())}
        for scenario, group in block.groupby("scenario"):
            entry = {}
            for metric in metrics:
                values = group[metric].to_numpy(dtype=float)
                entry[metric] = {"mean": float(values.mean()),
                                 "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
            entry["n"] = int(len(group))
            summary[config][scenario] = entry
    seed_count = int(frame.groupby("config")["seed"].nunique().min()) if len(frame) else 0
    lines = ["# 多种子 / 超参筛选汇总（test 划分）", "",
             "场景：`clean` 无缺失；`triple_30` 三模态各缺 30%；`single_*_30` 单模态缺 30%。",
             f"每个配置的种子数：{seed_count}", "",
             "## 汇总（均值 ± 标准差）", "",
             "| 配置 | 场景 | Accuracy | Macro-F1 | MAE | Pearson | CCC |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for config, block in summary.items():
        for scenario in SCENARIOS:
            entry = block.get(scenario)
            if not entry:
                continue
            lines.append(f"| {config} | {scenario} | "
                         f"{entry['accuracy']['mean']:.4f} ± {entry['accuracy']['std']:.4f} | "
                         f"**{entry['macro_f1']['mean']:.4f} ± {entry['macro_f1']['std']:.4f}** | "
                         f"{entry['mae']['mean']:.4f} ± {entry['mae']['std']:.4f} | "
                         f"{entry['pearson']['mean']:.4f} ± {entry['pearson']['std']:.4f} | "
                         f"{entry['ccc']['mean']:.4f} ± {entry['ccc']['std']:.4f} |")
    lines += ["", "## 关键对比（Macro-F1）", "",
              "| 配置 | clean | triple_30 | 平均最优 epoch |", "| --- | --- | --- | --- |"]
    for config, block in summary.items():
        def pick(scenario):
            item = block.get(scenario, {}).get("macro_f1")
            return f"{item['mean']:.4f} ± {item['std']:.4f}" if item else "n/a"
        lines.append(f"| {config} | {pick('clean')} | {pick('triple_30')} | {block['best_epoch_mean']:.1f} |")
    path.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=tuple(STUDIES), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--teacher-epochs", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--data-kind", choices=tuple(DATA_KINDS), default="aligned")
    parser.add_argument("--reuse", action="store_true", help="已存在 best.pt 的组合直接复用")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.output or (OUTPUT_ROOT / "_study" / args.study)
    root.mkdir(parents=True, exist_ok=True)
    print(f"研究={args.study} 种子={args.seeds} 轮数={args.epochs} patience={args.patience} "
          f"min_delta={args.min_delta}\n产物目录={root}", flush=True)

    records = train_all(args, root)
    pd.DataFrame(records).to_csv(root / "records.csv", index=False, encoding="utf-8-sig")
    if not records:
        raise SystemExit("没有成功训练的模型")
    frame = evaluate_all(args, root, records)
    frame.to_csv(root / "metrics.csv", index=False, encoding="utf-8-sig")
    summary = summarize(frame, root / "summary.md")
    (root / "summary.json").write_text(json.dumps({"study": args.study, "seeds": args.seeds,
                                                   "configs": {k: v for k, v in STUDIES[args.study].items()},
                                                   "summary": summary}, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("\n=== 关键对比（Macro-F1）===", flush=True)
    for config, block in summary.items():
        clean = block.get("clean", {}).get("macro_f1")
        triple = block.get("triple_30", {}).get("macro_f1")
        if clean and triple:
            print(f"{config}: clean {clean['mean']:.4f} ± {clean['std']:.4f} | "
                  f"triple_30 {triple['mean']:.4f} ± {triple['std']:.4f}", flush=True)
    print(f"已写入：{root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()

