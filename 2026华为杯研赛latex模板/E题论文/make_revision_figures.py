"""Redraw three comparison figures from recorded competition results.

The baseline and candidate figures describe one validation split. Text inputs
are different across systems, so these plots do not isolate architecture.
The independent test comparison has no confidence intervals or significance
claim. Run this after the existing figure scripts.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nature_plot_style import BLUE, GRAY, GRID, INK, ORANGE, TEAL, apply_style


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
OUT = PAPER / "figures"
OUT.mkdir(exist_ok=True)
apply_style()
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})

METRICS = (
    ("accuracy", "准确率", True),
    ("f1_macro", "宏平均 F1", True),
    ("mae", "平均绝对误差", False),
    ("pearson", "皮尔逊相关", True),
)
CANDIDATES = (
    ("Fixed aligned baseline", "固定对齐基线"),
    ("Length-aware resampled baseline", "长度感知重采样"),
    ("Bidirectional SSL fusion", "双向 GRU 融合"),
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    svg_path = OUT / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(OUT / f"{stem}.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_report_candidates() -> list[dict]:
    report = (ROOT / "q2_baseline" / "REPORT.md").read_text(encoding="utf-8")
    rows = []
    for source_name, label in CANDIDATES:
        line = next(
            line for line in report.splitlines() if line.startswith(f"| {source_name} |")
        )
        cells = [cell.strip().replace("*", "") for cell in line.strip("|").split("|")]
        values = list(map(float, cells[2:6]))
        rows.append(
            dict(
                model=label,
                source_model=source_name,
                accuracy=values[0],
                f1_macro=values[1],
                mae=values[2],
                pearson=values[3],
                source="q2_baseline/REPORT.md",
            )
        )
    final = json.loads(
        (ROOT / "q3_explainability" / "q3_aligned_text_bert_final" / "validation.json")
        .read_text(encoding="utf-8")
    )
    rows.append(
        dict(
            model="最终注意力门控",
            source_model="Final attention-gate model",
            **{key: final[key] for key, _, _ in METRICS},
            source="q3_explainability/q3_aligned_text_bert_final/validation.json",
        )
    )
    write_csv("source_candidate_metrics.csv", rows)
    return rows


def check_paper_values(rows: list[dict], test_final: dict) -> None:
    """Catch drift between plotted numbers and the manuscript tables."""
    paper = (PAPER / "main.tex").read_text(encoding="utf-8")
    for row in rows:
        line = next(
            line for line in paper.splitlines() if line.startswith(row["source_model"] + " &")
        )
        observed = [float(x) for x in re.findall(r"0\.\d{6}", line)[:4]]
        expected = [row[key] for key, _, _ in METRICS]
        if len(observed) != 4 or any(abs(a - b) > 6e-7 for a, b in zip(observed, expected)):
            raise ValueError(f"Manuscript metric mismatch: {row['source_model']}")
    for value in test_final.values():
        if f"{value:.6f}" not in paper:
            raise ValueError(f"Independent test result absent from manuscript: {value}")


def comparison_figure(rows: list[dict]) -> None:
    # Claim: the final system improves all four reported validation measures
    # over the fixed baseline. Archetype: quantitative grid.
    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.45), sharey=True)
    labels = [row["model"] for row in rows]
    y = np.arange(len(rows))
    for ax, (key, title, higher) in zip(axes, METRICS):
        values = np.array([row[key] for row in rows])
        lo, hi = values.min(), values.max()
        pad = max((hi - lo) * 0.42, 0.006)
        ax.set_xlim(lo - pad, hi + pad)
        ax.axvline(values[0], color=GRAY, lw=0.7, ls="--", zorder=0)
        for i, value in enumerate(values):
            color = TEAL if i == len(rows) - 1 else BLUE if i == 2 else GRAY
            ax.plot(value, i, "o", ms=5.7 if i == len(rows) - 1 else 4.5, color=color)
        ax.set_title(f"{title}\n{'↑' if higher else '↓'} 越{'高' if higher else '低'}越好")
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(labels)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.17, top=0.78, wspace=0.2)
    fig.text(
        0.18,
        0.025,
        "附件2验证集 n=728；各指标采用独立局部刻度；候选特征输入不同，历史残差方案仅列于正文表格。",
        fontsize=6.8,
        color=GRAY,
    )
    save(fig, "fig_model_comparison")


def performance_figure(baseline: dict, final: dict) -> None:
    # Claim: the final complete system has small but consistent validation
    # improvements. The paired points show exact levels without truncated bars.
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    for ax, (key, title, higher) in zip(axes, METRICS):
        old, new = baseline[key], final[key]
        lo, hi = sorted((old, new))
        pad = max((hi - lo) * 0.9, 0.004)
        ax.set_xlim(lo - pad, hi + pad)
        ax.plot([old, new], [0, 0], color=GRAY, lw=1.2, zorder=1)
        ax.scatter([old, new], [0, 0], c=[GRAY, TEAL], s=[42, 55], zorder=2)
        ax.annotate(f"{old:.4f}", (old, 0), xytext=(0, -17), textcoords="offset points",
                    ha="center", fontsize=7, color=GRAY)
        ax.annotate(f"{new:.4f}", (new, 0), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=7, color=TEAL, weight="bold")
        gain = new - old if higher else old - new
        ax.set_title(f"{title}\n定向改善 {gain:+.4f}")
        ax.set_ylim(-0.42, 0.42)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6.5)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.5)
    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.27, top=0.67, wspace=0.22)
    fig.text(0.06, 0.06, "● 固定基线", color=GRAY, fontsize=7)
    fig.text(0.24, 0.06, "● 最终模型", color=TEAL, fontsize=7)
    fig.text(
        0.45, 0.06, "同一验证划分 n=728；文本特征版本不同；横轴为各指标局部刻度。",
        color=GRAY, fontsize=6.8,
    )
    save(fig, "fig_main_performance")


def delta_figure(baseline: dict, final: dict) -> None:
    # Compare the selected validation model with the independently recorded
    # resampled test candidate. They use different feature organizations.
    # All values are in raw metric units; MAE is reversed so positive means
    # lower error. No p-value or confidence interval is implied.
    report = (ROOT / "q2_baseline" / "REPORT.md").read_text(encoding="utf-8")
    test_line = next(
        line for line in report.splitlines()
        if line.startswith("| Fixed aligned baseline | 0.650619")
    )
    test_baseline = dict(
        zip(
            (key for key, _, _ in METRICS),
            map(float, [cell.strip() for cell in test_line.strip("|").split("|")][1:5]),
        )
    )
    # Recorded independent test values for unaligned_resampled_50 are quoted
    # in REPORT.md and cross-checked against the manuscript table.
    test_candidate = dict(accuracy=0.664374, f1_macro=0.617797,
                          mae=0.649980, pearson=0.655419)
    check_paper_values([baseline, final], test_candidate)
    source_rows = []
    for split, base, outcome in (
        ("validation", baseline, final),
        ("test", test_baseline, test_candidate),
    ):
        for key, label, higher in METRICS:
            delta = outcome[key] - base[key] if higher else base[key] - outcome[key]
            source_rows.append(
                dict(split=split, metric=key, metric_label=label, baseline=base[key],
                     final=outcome[key], directional_delta=delta,
                     source=("q2_baseline/REPORT.md + q3 validation.json"
                             if split == "validation" else "q2_baseline/REPORT.md"))
            )
    write_csv("source_metric_deltas.csv", source_rows)
    fig, ax = plt.subplots(figsize=(7.0, 2.75))
    y = np.arange(4)
    ax.axvline(0, color=INK, lw=0.8)
    for split, offset, marker in (("validation", -0.16, "o"), ("test", 0.16, "D")):
        rows = [row for row in source_rows if row["split"] == split]
        for i, row in enumerate(rows):
            value = row["directional_delta"]
            color = TEAL if value >= 0 else ORANGE
            ax.plot(value, i + offset, marker=marker, ms=6, color=color, zorder=3)
            ax.annotate(f"{value:+.4f}", (value, i + offset),
                        xytext=(5 if value >= 0 else -5, 0), textcoords="offset points",
                        ha="left" if value >= 0 else "right", va="center",
                        fontsize=7, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label, _ in METRICS])
    ax.invert_yaxis()
    ax.set_xlim(-0.027, 0.038)
    ax.set_xlabel("定向差值（原始指标单位；正值表示改善）")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.plot([], [], "o", color=GRAY, label="验证集：模型选择")
    ax.plot([], [], "D", color=GRAY, label="测试集：重采样候选")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
              fontsize=7, ncol=2, handletextpad=0.3)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.24, top=0.83)
    save(fig, "fig_metric_delta")


def main() -> None:
    rows = parse_report_candidates()
    check_paper_values(rows, dict(accuracy=0.664374, f1_macro=0.617797,
                                  mae=0.649980, pearson=0.655419))
    comparison_figure(rows)
    performance_figure(rows[0], rows[-1])
    delta_figure(rows[0], rows[-1])
    print("Revised three figures and wrote two source-data CSV files.")


if __name__ == "__main__":
    main()
