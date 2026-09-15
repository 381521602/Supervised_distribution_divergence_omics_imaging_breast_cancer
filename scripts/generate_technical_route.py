#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a clean technical-route/method-overview diagram."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "paper_figures" / "fig1_technical_route.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text, fc, ec, fs=9, text_color="white"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=text_color,
        wrap=True,
    )


def arrow(ax, p1, p2, color="#444444", style="-|>"):
    a = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=1.2,
        color=color,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(13.0, 6.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # First row
    box(ax, 0.20, 3.55, 2.20, 1.55, "数据获取\nTCGA-BRCA\nmRNA / CNV / miRNA\n临床与随访", "#1F4D78", "#1F4D78")
    box(ax, 2.80, 3.55, 2.35, 1.55, "样本对齐与标签\nPAM50 四分类\n总生存事件 / 时间", "#2E74B5", "#2E74B5")
    box(ax, 5.55, 3.55, 2.35, 1.55, "特征筛选\n低方差 / F 值\nL1 / NSRE", "#2E74B5", "#2E74B5")
    box(ax, 8.30, 3.55, 2.35, 1.55, "NSRE 图像化\n排序 + 中心向外螺旋\n灰度图 / 功能类别图", "#0E7C86", "#0E7C86")
    box(ax, 10.90, 3.55, 1.90, 1.55, "建模\nFullSizeCNN\nML / MLP / CoxPH", "#6B46C1", "#6B46C1")

    # Second row
    box(ax, 1.20, 0.75, 2.20, 1.45, "单组学预测\nPAM50 四分类\n生存风险", "#B42318", "#B42318")
    box(ax, 3.80, 0.75, 2.35, 1.45, "多组学融合\nConcat / Gated / LateAvg\nStacking / Transformer", "#B42318", "#B42318")
    box(ax, 6.55, 0.75, 2.25, 1.45, "统一评估\n5 折交叉验证\nAccuracy / Macro-F1\nROC AUC / C-index", "#D97706", "#D97706")
    box(ax, 9.20, 0.75, 2.45, 1.45, "可解释分析\nSHAP / CNN saliency\n关键基因与通路", "#2F855A", "#2F855A")

    # First-row arrows
    arrow(ax, (2.40, 4.32), (2.78, 4.32))
    arrow(ax, (5.15, 4.32), (5.53, 4.32))
    arrow(ax, (7.90, 4.32), (8.28, 4.32))
    arrow(ax, (10.65, 4.32), (10.88, 4.32))

    # Row transition arrows
    arrow(ax, (2.30, 3.53), (2.30, 2.22), color="#666666")
    arrow(ax, (4.98, 3.53), (4.98, 2.22), color="#666666")
    arrow(ax, (7.68, 3.53), (7.68, 2.22), color="#666666")
    arrow(ax, (10.42, 3.53), (10.42, 2.22), color="#666666")

    # Second-row arrows
    arrow(ax, (3.40, 1.47), (3.78, 1.47))
    arrow(ax, (6.15, 1.47), (6.53, 1.47))
    arrow(ax, (8.80, 1.47), (9.18, 1.47))

    # Dashed feedback from interpretability to NSRE imaging / feature selection
    arrow(ax, (10.42, 0.75), (9.48, 3.53), color="#888888", style="-|>")
    ax.text(
        9.50,
        1.75,
        "反馈关键基因与通路",
        fontsize=8,
        color="#555555",
        ha="center",
        va="center",
    )

    ax.text(6.5, 5.72, "NSRE 引导的组学图像化与多组学融合技术路线",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#0B2545")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.005)
    fig.savefig(OUT, dpi=220, facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
