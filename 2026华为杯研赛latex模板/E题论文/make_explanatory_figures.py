"""Rebuild three evidence figures from the recorded CSV files, without inference.

Run after the original figure scripts.  The figure labels are in Chinese and
the numerical annotations come directly from the committed source CSV files.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from nature_plot_style import BLUE, GRAY, GRID, INK, ORANGE, TEAL, apply_style


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
OUT = PAPER / "figures"
apply_style()
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titleweight": "normal"})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 350})):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    svg = OUT / f"{stem}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def confusion() -> None:
    rows = read_csv(OUT / "source_confusion_matrix.csv")
    labels = ["消极", "中性", "积极"]
    matrix = np.array([[int(row[k]) for k in ("Negative", "Neutral", "Positive")] for row in rows])
    assert matrix.shape == (3, 3) and matrix.sum() == 728
    row_pct = matrix / matrix.sum(axis=1, keepdims=True) * 100
    cmap = LinearSegmentedColormap.from_list("paper_blue", ["#F5F8FB", "#91BAD5", BLUE])
    fig, ax = plt.subplots(figsize=(5.0, 3.55))
    ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=80, aspect="equal")
    for i in range(3):
        for j in range(3):
            color = "white" if row_pct[i, j] > 60 else INK
            ax.text(j, i - .08, str(matrix[i, j]), ha="center", va="center", color=color,
                    fontsize=14)
            ax.text(j, i + .23, f"{row_pct[i, j]:.1f}%", ha="center", va="center",
                    color=color, fontsize=8.5)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)
    fig.text(.52, .01, "每格上方为样本数，下方为该真实类别内占比", ha="center", color=GRAY, fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    save(fig, "fig_validation_confusion")


def missing_heatmap() -> None:
    rows = read_csv(OUT / "source_missing_scenarios.csv")
    modalities = [("text", "文本"), ("audio", "语音"), ("vision", "视觉")]
    positions = [("start", "起始"), ("middle", "中段"), ("end", "末段")]
    windows = [5, 15, 25]
    index = {(r["modality"], r["position"], int(r["masked_windows"])): 100 * float(r["delta_f1"]) for r in rows}
    assert len(index) == 27
    fig, ax = plt.subplots(figsize=(5.3, 4.45))
    row_labels = []
    for m, (modality, title) in enumerate(modalities):
        for p, (position, position_name) in enumerate(positions):
            i = m * 3 + p
            row_labels.append(f"{title} · {position_name}")
            for j, width in enumerate(windows):
                value = index[(modality, position, width)]
                # Hue shows direction.  Saturation is local to each sign; the
                # printed percentage-point value carries the exact magnitude.
                strength = min(abs(value) / (16 if value < 0 else 1.6), 1)
                rgb = np.array(matplotlib.colors.to_rgb(ORANGE if value < 0 else TEAL))
                face = tuple((1 - .12 - .72 * strength) * np.ones(3) + (.12 + .72 * strength) * rgb)
                if abs(value) < .05:
                    face = "#F5F7F9"
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=face,
                                           edgecolor="white", linewidth=1.6))
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center",
                        color="white" if strength > .70 else INK,
                        fontsize=9)
    for boundary in (2.5, 5.5):
        ax.axhline(boundary, color=GRAY, linewidth=.8)
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(8.5, -.5)
    ax.set_xticks(range(3), [str(x) for x in windows])
    ax.set_yticks(range(9), row_labels)
    ax.set_xlabel("被遮挡窗口数")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.51, .005, "Macro-F1 变化（百分点）：红色下降，绿色上升", ha="center", color=GRAY, fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    save(fig, "fig_missing_heatmap")


def attachment4() -> None:
    rows = read_csv(ROOT / "q3_explainability" / "q3_aligned_text_bert_final" / "attachment4" / "q3_predictions.csv")
    assert len(rows) == 20
    names = [r["predicted_class"] for r in rows]
    values = np.array([float(r["predicted_strength"]) for r in rows])
    assert Counter(names) == {"Negative": 8, "Neutral": 3, "Positive": 9}
    colors = {"Negative": ORANGE, "Neutral": GRAY, "Positive": TEAL}
    labels = {"Negative": "消极", "Neutral": "中性", "Positive": "积极"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.45),
                                   gridspec_kw={"width_ratios": [.8, 1.7]})
    counts = Counter(names)
    for x, name in enumerate(("Negative", "Neutral", "Positive")):
        ax1.bar(x, counts[name], width=.58, color=colors[name])
        ax1.text(x, counts[name] + .2, str(counts[name]), ha="center", color=INK, fontsize=9)
    ax1.set_xticks(range(3), [labels[x] for x in ("Negative", "Neutral", "Positive")])
    ax1.set_ylim(0, 10.5)
    ax1.set_ylabel("样本数")
    ax1.yaxis.grid(True, alpha=.6)
    ax1.set_axisbelow(True)

    x = np.arange(1, 21)
    ax2.axhline(0, color=GRAY, linewidth=.8)
    ax2.axhline(values.mean(), color=BLUE, linestyle="--", linewidth=1,
                label=f"平均值 {values.mean():+.3f}")
    for name in ("Negative", "Neutral", "Positive"):
        selected = np.array([i for i, item in enumerate(names) if item == name])
        ax2.scatter(x[selected], values[selected], s=44, color=colors[name],
                    label=labels[name], zorder=3)
    ax2.set_xlim(.3, 20.7)
    ax2.set_ylim(-2.35, 1.5)
    ax2.set_xticks([1, 5, 10, 15, 20])
    ax2.set_xlabel("附件4样本序号")
    ax2.set_ylabel("预测情感强度")
    ax2.yaxis.grid(True, alpha=.6)
    ax2.set_axisbelow(True)
    ax2.legend(loc="lower left", ncol=2, fontsize=7.5, handletextpad=.3,
               columnspacing=.7)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_attachment4")


if __name__ == "__main__":
    confusion()
    missing_heatmap()
    attachment4()
