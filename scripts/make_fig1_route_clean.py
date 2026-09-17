#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a clean, publication-style technical route diagram (no result figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT = Path(r"E:\组学数据图像化")
OUT = ROOT / "data" / "paper_figures" / "fig1_method_overview_clean.png"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def module(ax, x, y, w, h, no, title, items, accent, dark):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.12,rounding_size=1.1",
            linewidth=1.4,
            edgecolor=dark,
            facecolor="white",
            zorder=2,
        )
    )
    # Accent strip at top
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 3.4),
            w,
            3.4,
            boxstyle="round,pad=0.12,rounding_size=1.1",
            linewidth=0,
            edgecolor="none",
            facecolor=accent,
            zorder=3,
        )
    )
    ax.add_patch(
        Circle((x + 2.1, y + h - 1.7), 1.15, facecolor="white", edgecolor=dark, linewidth=1.3, zorder=4)
    )
    ax.text(x + 2.1, y + h - 1.7, no, ha="center", va="center", fontsize=11, fontweight="bold", color=dark, zorder=5)
    ax.text(x + 3.7, y + h - 1.7, title, ha="left", va="center", fontsize=14.5, fontweight="bold", color=dark, zorder=5)

    yy = y + h - 5.0
    for item in items:
        ax.text(x + 1.1, yy, "•  " + item, ha="left", va="center", fontsize=10.8, color="#25344A", zorder=5)
        yy -= 2.45


def arrow(ax, p0, p1, color="#3E7CAA", rad=0.0, connectionstyle=None):
    kw = dict(
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=2.0,
        color=color,
        shrinkA=2,
        shrinkB=2,
        zorder=1,
    )
    if connectionstyle:
        kw["connectionstyle"] = connectionstyle
    else:
        kw["connectionstyle"] = f"arc3,rad={rad}"
    ax.add_patch(FancyArrowPatch(p0, p1, **kw))


def main() -> None:
    fig = plt.figure(figsize=(13.2, 7.1), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 132)
    ax.set_ylim(0, 71)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")

    # Title block
    ax.text(66, 67.6, "JSD-Guided Omics Imaging and Multi-Omics Fusion for Breast Cancer",
            ha="center", va="center", fontsize=16.5, fontweight="bold", color="#173A5E")
    ax.text(66, 64.2, "基于 JSD 引导组学图像化与多组学融合的乳腺癌分子分型及生存预测 — 方法路线",
            ha="center", va="center", fontsize=11.2, color="#5E6E83")
    ax.plot([3, 129], [61.5, 61.5], color="#D7E2EC", linewidth=1.2)

    # Row 1
    y = 43.5
    h = 13.5
    w = 38.0
    gap_x = 8.0
    x1 = 2.0
    x2 = x1 + w + gap_x
    x3 = x2 + w + gap_x

    module(ax, x1, y, w, h, "1", "数据获取与样本对齐", ["TCGA-BRCA 多组学数据", "mRNA / CNV / miRNA", "统一交集样本清单"], "#2E6FA6", "#24547F")
    module(ax, x2, y, w, h, "2", "临床标签定义", ["PAM50 四分类标签", "总生存 / 事件状态", "OS 时间与删失信息"], "#348FB3", "#2A718F")
    module(ax, x3, y, w, h, "3", "无泄漏特征筛选", ["低方差 / ANOVA-F", "L1 / LASSO", "JSD 及组合扩增"], "#60A5C7", "#4C84A0")

    arrow(ax, (x1 + w, y + h / 2), (x2, y + h / 2))
    arrow(ax, (x2 + w, y + h / 2), (x3, y + h / 2))

    # Row 2
    y2 = 16.0
    module(ax, x1, y2, w, h, "4", "JSD 组学图像化", ["重要性降序排序", "中心向外螺旋填充", "灰度 / 功能 / JSD 图"], "#81BBC8", "#6799A5")
    module(ax, x2, y2, w, h, "5", "单组学建模", ["LR / RF / GBM / SVC / KNN", "MLP", "FullSizeCNN"], "#F2A45A", "#C48649")
    module(ax, x3, y2, w, h, "6", "多组学融合与评估", ["拼接 / 注意力 / Stacking", "5 折交叉验证", "Acc / Macro-F1 / AUC / C-index"], "#D86C6C", "#AD5555")

    arrow(ax, (x3 + w / 2, y), (x3 + w / 2, y2 + h))
    arrow(ax, (x3 + w / 2, y2 + h), (x2 + w / 2, y2 + h))
    arrow(ax, (x2 + w, y2 + h / 2), (x1 + w, y2 + h / 2), rad=-0.08)

    # Bottom note for interpretability
    ax.text(66, 5.0, "下游可解释性：SHAP · Grad-CAM · 关键因子与通路挖掘",
            ha="center", va="center", fontsize=11.0, fontweight="bold", color="#A23E3E",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FBF0F0", edgecolor="#D9A3A3", linewidth=1.0))

    ax.text(66, 1.7, "图 1. 研究方法总览。所有特征筛选与模型拟合均在 5 折交叉验证训练折内完成，测试折不参与任何预处理与参数选择。",
            ha="center", va="center", fontsize=8.2, color="#6E7A8A")

    fig.savefig(OUT, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
