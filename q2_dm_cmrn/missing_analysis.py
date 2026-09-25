"""缺失规律分析：网格化考察“缺失类型 / 位置 / 时长 / 缺失率”对预测性能的影响。

示例：
    python missing_analysis.py --checkpoint outputs/dm_cmrn/best.pt --split valid
    python missing_analysis.py --checkpoint outputs/dm_cmrn/best.pt --grids type ratio
产出 outputs/missing_analysis/：
    missing_grid.csv        全部网格结果
    missing_analysis.md     分维度汇总表（可直接用于论文）
    curve_ratio.png         性能随缺失率变化曲线（需要 matplotlib）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate import evaluate_scenarios  # noqa: E402
from src.data import Scenario  # noqa: E402
from src.engine import load_checkpoint, load_prepared_splits, resolve_device, set_seed  # noqa: E402
from src.paths import DATA_KINDS, MODALITIES, OUTPUT_ROOT  # noqa: E402

TYPES = ["text", "audio", "vision", "text+audio", "text+vision", "audio+vision", "text+audio+vision"]
POSITIONS = ["start", "middle", "end", "random"]
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]
DURATIONS = [2, 4, 8]


def build_grid(grids: list[str]) -> dict:
    """根据选定的维度生成“场景名 → Scenario”字典。"""
    scenarios = {"clean": Scenario(mode="none")}
    if "type" in grids:
        for mode in TYPES:
            scenarios[f"type|{mode}|r30"] = Scenario(mode=mode, position="random", ratio=0.3)
    if "position" in grids:
        for position in POSITIONS:
            for mode in ("text", "audio", "vision"):
                scenarios[f"position|{position}|{mode}"] = Scenario(mode=mode, position=position, ratio=0.3)
    if "ratio" in grids:
        for ratio in RATIOS:
            scenarios[f"ratio|all|{ratio}"] = Scenario(mode="text+audio+vision", position="random",
                                                       ratio=ratio)
            for mode in MODALITIES:
                scenarios[f"ratio|{mode}|{ratio}"] = Scenario(mode=mode, position="random", ratio=ratio)
    if "duration" in grids:
        for duration in DURATIONS:
            for mode in MODALITIES:
                scenarios[f"duration|{mode}|{duration}w"] = Scenario(mode=mode, position="middle",
                                                                     ratio=0.0, duration=duration)
    return scenarios


def parse_scenario_name(name: str) -> dict:
    parts = name.split("|")
    return {"grid": parts[0], "detail": "|".join(parts[1:])}


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    lines = ["# 局部模态缺失规律分析", "",
             "指标：Accuracy / Macro-F1（分类），MAE / Pearson / CCC（回归，强度范围 -3~3）。",
             "缺失定义：在**有效时间轴内部**制造连续零区间（与附件3 的局部缺失语义一致）。", ""]
    clean = frame[frame["grid"] == "clean"]
    if len(clean):
        row = clean.iloc[0]
        lines += ["## 无缺失对照", "",
                  f"- 干净输入：Acc {row['accuracy']:.4f}，Macro-F1 {row['macro_f1']:.4f}，"
                  f"MAE {row['mae']:.4f}，Pearson {row['pearson']:.4f}", ""]
    for grid in frame["grid"].unique():
        if grid == "clean":
            continue
        block = frame[frame["grid"] == grid]
        lines += [f"## 维度：{grid}", "",
                  "| 场景 | Accuracy | Macro-F1 | MAE | Pearson | CCC |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for _, row in block.iterrows():
            lines.append(f"| {row['detail']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
                         f"{row['mae']:.4f} | {row['pearson']:.4f} | {row['ccc']:.4f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_ratio_curve(frame: pd.DataFrame, path: Path) -> bool:
    """绘制“性能随缺失率变化”曲线（缺少 matplotlib 时安全跳过）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False
    block = frame[frame["grid"] == "ratio"].copy()
    if not len(block):
        return False
    block["target"] = block["detail"].str.split("|").str[0]
    block["ratio"] = block["detail"].str.split("|").str[1].astype(float)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for target, group in block.groupby("target"):
        group = group.sort_values("ratio")
        axes[0].plot(group["ratio"], group["macro_f1"], marker="o", label=target)
        axes[1].plot(group["ratio"], group["mae"], marker="s", label=target)
    axes[0].set_xlabel("missing ratio"); axes[0].set_ylabel("Macro-F1"); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("missing ratio"); axes[1].set_ylabel("MAE"); axes[1].grid(alpha=0.3)
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--data-kind", choices=tuple(DATA_KINDS), default="aligned")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--grids", nargs="+", choices=("type", "position", "ratio", "duration"),
                        default=["type", "position", "ratio", "duration"])
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "missing_analysis")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    data_path = args.data or DATA_KINDS[args.data_kind]
    bundle = load_prepared_splits(Path(data_path), need_test=args.split == "test")
    scenarios = build_grid(args.grids)
    print(f"模型={payload['model_name']}，场景数={len(scenarios)}，划分={args.split}", flush=True)
    results = evaluate_scenarios(model, bundle["prepared"][args.split], device, args.batch_size,
                                 scenarios, args.seed)
    rows = []
    for name, metrics in results.items():
        rows.append({"scenario": name, **parse_scenario_name(name),
                     **{key: value for key, value in metrics.items() if key != "prediction_distribution"}})
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "missing_grid.csv", index=False, encoding="utf-8-sig")
    write_markdown(frame, args.output / "missing_analysis.md")
    plotted = plot_ratio_curve(frame, args.output / "curve_ratio.png")
    (args.output / "missing_analysis.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "model": payload["model_name"], "split": args.split,
         "grids": args.grids, "scenarios": len(scenarios), "curve_plotted": bool(plotted)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame[["scenario", "macro_f1", "mae", "pearson"]].to_string(index=False), flush=True)
    print(f"已写入：{args.output}", flush=True)


if __name__ == "__main__":
    main()

