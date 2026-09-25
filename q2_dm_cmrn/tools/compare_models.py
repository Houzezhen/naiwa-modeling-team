"""多模型对比：对同一目录下所有检查点跑同一套缺失场景，输出对比表。

用法：
    python tools/compare_models.py --root outputs --split test
    python tools/compare_models.py --root outputs --split test --models dm_cmrn teacher early misa
产出：<root>/comparison_<split>.csv 与 <root>/comparison_<split>.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from evaluate import default_scenarios, evaluate_scenarios  # noqa: E402
from src.engine import load_checkpoint, load_prepared_splits, resolve_device, set_seed  # noqa: E402
from src.paths import DATA_KINDS, OUTPUT_ROOT  # noqa: E402

# 与文档表格一致的两列重点场景
KEY_SCENARIOS = ("clean", "triple_30")


def discover(root: Path, names: list[str] | None) -> list[Path]:
    if names:
        return [root / name / "best.pt" for name in names if (root / name / "best.pt").is_file()]
    return sorted(path for path in root.glob("*/best.pt") if not path.parent.name.startswith("_"))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT, help="包含各模型子目录的根目录")
    parser.add_argument("--models", nargs="+", default=None, help="指定子目录名；默认扫描全部")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--data-kind", choices=tuple(DATA_KINDS), default="aligned")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--all-scenarios", action="store_true",
                        help="评估全部 8 个缺失场景；默认只评估 clean 与 triple_30（更快）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    data_path = args.data or DATA_KINDS[args.data_kind]
    checkpoints = discover(args.root, args.models)
    if not checkpoints:
        raise SystemExit(f"在 {args.root} 下没有找到任何 */best.pt")
    bundle = load_prepared_splits(Path(data_path), need_test=args.split == "test")
    scenarios = default_scenarios()
    if not args.all_scenarios:
        scenarios = {name: scenarios[name] for name in KEY_SCENARIOS if name in scenarios}

    rows, records = [], {}
    for checkpoint in checkpoints:
        model, payload = load_checkpoint(checkpoint, device)
        results = evaluate_scenarios(model, bundle["prepared"][args.split], device, args.batch_size,
                                     scenarios, args.seed)
        records[checkpoint.parent.name] = results
        row = {"model": checkpoint.parent.name, "config": str(payload.get("model_kwargs")),
               "params": sum(p.numel() for p in model.parameters())}
        for name in KEY_SCENARIOS:
            if name in results:
                metrics = results[name]
                row.update({f"{name}_acc": metrics["accuracy"], f"{name}_f1": metrics["macro_f1"],
                            f"{name}_mae": metrics["mae"], f"{name}_pearson": metrics["pearson"],
                            f"{name}_ccc": metrics["ccc"]})
        rows.append(row)
        print(f"{row['model']}: clean F1={row.get('clean_f1', float('nan')):.4f} "
              f"triple_30 F1={row.get('triple_30_f1', float('nan')):.4f} "
              f"MAE={row.get('clean_mae', float('nan')):.4f}", flush=True)

    frame = pd.DataFrame(rows)
    args.root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.root / f"comparison_{args.split}.csv", index=False, encoding="utf-8-sig")
    lines = [f"# 多模型对比（{args.split} 划分，{len(checkpoints)} 个检查点）", "",
             "场景：`clean` = 无缺失；`triple_30` = 三模态各在有效段内随机缺失 30%。", "",
             "| 模型 | 参数量 | clean Acc | clean Macro-F1 | clean MAE | clean Pearson | "
             "triple_30 Macro-F1 | triple_30 MAE |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['params']} | {row.get('clean_acc', float('nan')):.4f} | "
                     f"{row.get('clean_f1', float('nan')):.4f} | {row.get('clean_mae', float('nan')):.4f} | "
                     f"{row.get('clean_pearson', float('nan')):.4f} | "
                     f"{row.get('triple_30_f1', float('nan')):.4f} | "
                     f"{row.get('triple_30_mae', float('nan')):.4f} |")
    (args.root / f"comparison_{args.split}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写入：{args.root / f'comparison_{args.split}.csv'} 与 .md", flush=True)


if __name__ == "__main__":
    main()
