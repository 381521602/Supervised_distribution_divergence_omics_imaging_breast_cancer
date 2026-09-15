#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure for the scheme-2 category-masking interpretability workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data/images/PAM50/mRNA_reorder/scheme2_function_center"
OUT = ROOT / "data/interpretability/figures/fig_scheme2_masking_workflow_en.png"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

CATEGORY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]
CATEGORY_NAMES = {
    0: "Development / Epithelium",
    1: "Signaling / Transport",
    2: "Hormone / Metabolism",
    3: "Immune / Inflammation",
    4: "Cell cycle / Proliferation",
    5: "Other",
}


def add_box(ax, x, y, w, h, text, color, text_color="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.15,rounding_size=0.8",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, fontweight="bold", color=text_color)


def add_arrow(ax, p0, p1, color="#5E6E83"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2,
            color=color,
        )
    )


def main():
    gray = np.load(IMG / "images.npy").astype(np.float32)
    cat_grid = np.load(IMG / "category_grid.npy").astype(int)
    sample_idx = 0
    original = gray[sample_idx, 0]
    masked_other = original.copy()
    masked_other[cat_grid == 5] = 0.0

    fig = plt.figure(figsize=(13, 7.4), dpi=300)
    fig.patch.set_facecolor("#F7FAFC")
    gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.25], hspace=0.22, wspace=0.08, left=0.04, right=0.74, top=0.95, bottom=0.04)

    ax_flow = fig.add_subplot(gs[0, :])
    ax_flow.set_xlim(0, 100)
    ax_flow.set_ylim(0, 14)
    ax_flow.axis("off")

    steps = [
        ("Scheme-2 spiral map", "#2E6FA6"),
        ("Functional category map", "#348FB3"),
        ("Mask selected category", "#D86C6C"),
        ("FullSizeCNN", "#F2A45A"),
        ("PAM50 / Survival", "#5F9E6E"),
    ]
    box_w = 16.5
    box_h = 4.2
    x_start = 2
    y = 4.6
    gap = 3.0
    xs = []
    for i, (text, color) in enumerate(steps):
        x = x_start + i * (box_w + gap)
        xs.append(x)
        add_box(ax_flow, x, y, box_w, box_h, text, color)
        if i < len(steps) - 1:
            add_arrow(ax_flow, (x + box_w, y + box_h / 2), (x + box_w + gap, y + box_h / 2))

    ax_flow.text(50, 1.0, "Before masking: full scheme-2 image; after masking: set pixels of target functional category to zero and compare model performance", ha="center", va="center", fontsize=10, color="#33465D")

    cmap = ListedColormap(CATEGORY_COLORS)
    ax0 = fig.add_subplot(gs[1, 0])
    ax0.imshow(original, cmap="gray", aspect="auto")
    ax0.set_title("A. Before masking: full expression map", fontsize=11, fontweight="bold")
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[1, 1])
    im1 = ax1.imshow(cat_grid, cmap=cmap, vmin=0, vmax=5, aspect="auto")
    ax1.set_title("B. Functional category map", fontsize=11, fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=CATEGORY_COLORS[i], markersize=10, label=CATEGORY_NAMES[i]) for i in sorted(set(np.unique(cat_grid).tolist()))]
    ax1.legend(handles=handles, bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8, frameon=False, borderaxespad=0)

    ax2 = fig.add_subplot(gs[1, 2])
    ax2.imshow(masked_other, cmap="gray", aspect="auto")
    ax2.set_title("C. After masking: Other genes removed", fontsize=11, fontweight="bold")
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.suptitle("Scheme-2 mRNA category-masking interpretability workflow", fontsize=13, fontweight="bold")
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
