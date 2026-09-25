"""一键串跑：审计 → Baselines → Teacher → DM-CMRN → 消融 → 评估 → 缺失规律 → 附件3 推理。

示例：
    python tools/run_all.py --smoke                 # 冒烟：小样本 1 轮，验证全链路可跑
    python tools/run_all.py --steps audit baselines teacher main evaluate
    python tools/run_all.py                         # 正式：全部步骤（GPU 建议）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from src.models import ABLATION_VARIANTS  # noqa: E402
from src.paths import OUTPUT_ROOT  # noqa: E402

ALL_STEPS = ("audit", "baselines", "teacher", "main", "ablations", "evaluate", "missing",
             "attachment3", "compare")

# 这些步骤读取主模型检查点，若不存在则跳过而不是报错（例如只跑 --smoke --steps audit）
DEPENDENT_STEPS = ("evaluate", "missing", "attachment3")
# 这些步骤需要根目录下至少存在一个 */best.pt
NEEDS_MODELS = ("compare",)


def resolve_output_root(args) -> Path:
    """产物根目录：--smoke 时自动隔离到 outputs/_smoke，避免覆盖正式训练结果。"""
    if getattr(args, "output_root", None):
        return Path(args.output_root)
    return OUTPUT_ROOT / "_smoke" if args.smoke else OUTPUT_ROOT


def python_cmd(*parts: str) -> list[str]:
    return [sys.executable, "-u", *parts]


def build_commands(args) -> list[tuple[str, list[str]]]:
    smoke = ["--max-train", str(args.smoke_train), "--max-valid", str(args.smoke_valid),
             "--epochs", "1", "--patience", "1"] if args.smoke else []
    base = ["--batch-size", str(args.batch_size), "--hidden", str(args.hidden),
            "--layers", str(args.layers), "--dropout", str(args.dropout), "--device", args.device]
    common = base + smoke
    epochs = [] if args.smoke else ["--epochs", str(args.epochs), "--patience", str(args.patience)]
    root = resolve_output_root(args)
    teacher_out = root / "teacher"
    commands = {
        "audit": [python_cmd(str(HERE / "tools" / "audit_data.py"))],
        "baselines": [python_cmd(str(HERE / "train.py"), "--model", model, *common, *epochs,
                                 "--output", str(root / model))
                      for model in ("early", "cross_attn", "misa")],
        "teacher": [python_cmd(str(HERE / "train.py"), "--model", "cross_attn", "--stage", "teacher",
                               "--clean-only", *common, *epochs, "--output", str(teacher_out))],
        "main": [python_cmd(str(HERE / "train.py"), "--model", "dm_cmrn", "--stage", "student",
                            "--teacher", str(teacher_out / "best.pt"), "--rec-weight", str(args.rec_weight),
                            "--distill-weight", str(args.distill_weight), *common, *epochs,
                            "--output", str(root / "dm_cmrn"))],
        "ablations": [python_cmd(str(HERE / "train.py"), "--model", "dm_cmrn", "--variant", variant,
                                 "--teacher", str(teacher_out / "best.pt"),
                                 "--rec-weight", str(args.rec_weight), *common, *epochs,
                                 "--output", str(root / variant))
                      for variant in ABLATION_VARIANTS if variant != "dm_cmrn_full"],
        "evaluate": [python_cmd(str(HERE / "evaluate.py"), "--checkpoint", str(root / "dm_cmrn" / "best.pt"),
                                "--split", "test", "--batch-size", str(args.batch_size), "--device", args.device)],
        "missing": [python_cmd(str(HERE / "missing_analysis.py"),
                               "--checkpoint", str(root / "dm_cmrn" / "best.pt"),
                               "--split", "valid", "--batch-size", str(args.batch_size), "--device", args.device)],
        "attachment3": [python_cmd(str(HERE / "infer_attachment3.py"),
                                   "--checkpoint", str(root / "dm_cmrn" / "best.pt"),
                                   "--device", "cpu")],
        "compare": [python_cmd(str(HERE / "tools" / "compare_models.py"), "--root", str(root),
                               "--split", args.compare_split, "--batch-size", str(args.batch_size),
                               "--device", args.device)],
    }
    result = []
    for step in args.steps:
        for command in commands[step]:
            result.append((step, command))
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=list(ALL_STEPS))
    parser.add_argument("--smoke", action="store_true", help="小样本 1 轮，快速验证链路")
    parser.add_argument("--smoke-train", type=int, default=192)
    parser.add_argument("--smoke-valid", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--rec-weight", type=float, default=0.3)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compare-split", choices=("valid", "test"), default="test",
                        help="compare 步骤使用的划分（默认 test）")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="显式指定产物根目录；--smoke 时默认隔离到 outputs/_smoke")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    args = parser.parse_args(argv)

    root = resolve_output_root(args)
    checkpoint = root / "dm_cmrn" / "best.pt"
    if args.smoke:
        print(f"[冒烟模式] 所有产物写入 {root}（不会覆盖正式输出 {OUTPUT_ROOT}）", flush=True)
    commands = build_commands(args)
    failures = []
    for step, command in commands:
        if step in DEPENDENT_STEPS and not checkpoint.is_file():
            print(f"\n=== [{step}] 跳过：缺少主模型检查点 {checkpoint}（请先运行 main 步骤）", flush=True)
            continue
        if step in NEEDS_MODELS and not any(root.glob("*/best.pt")):
            print(f"\n=== [{step}] 跳过：{root} 下没有找到任何 */best.pt（请先训练模型）", flush=True)
            continue
        print(f"\n=== [{step}] {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=str(HERE))
        if completed.returncode != 0:
            failures.append((step, completed.returncode))
            print(f"步骤 {step} 失败（code={completed.returncode}），继续后续步骤", flush=True)
    print("\n=== 汇总 ===")
    if failures:
        print("失败步骤：", failures, flush=True)
        raise SystemExit(1)
    print("全部步骤完成", flush=True)


if __name__ == "__main__":
    main()
