#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flow diagram for the scheme-2 category-masking interpretability analysis."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data/images/PAM50/mRNA_reorder/scheme2_function_center"
OUT = ROOT / "data/interpretability/figures/fig_scheme2_masking_flow_en_v2.png"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

CATEGORY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]


def box(ax, x, y, w, h, text, color, text_color="white", fontsize=9.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.12,rounding_size=0.55",
            linewidth=1.1,
            edgecolor=color,
            facecolor=color,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=text_color)


def arrow(ax, p0, p1, color="#5E6E83"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2,
            color=color,
        )
    )


def main():
    gray = np.load(IMG / "images.npy").astype(np.float32)
    cat_grid = np.load(IMG / "category_grid.npy").astype(int)
    original = gray[0, 0]
    masked = original.copy()
    masked[cat_grid == 5] = 0.0

    fig = plt.figure(figsize=(13.2, 6.0), dpi=300)
    fig.patch.set_facecolor("#F7FAFC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    steps = [
        ("Scheme-2\nspiral image", "#2E6FA6"),
        ("Functional\ncategory map", "#348FB3"),
        ("Select target\ncategory", "#60A5C7"),
        ("Mask region\n(set to 0)", "#D86C6C"),
        ("FullSizeCNN", "#F2A45A"),
        ("PAM50 /\nSurvival", "#5F9E6E"),
    ]
    y = 41
    box_h = 13
    box_w = 13.8
    gap = 2.6
    x = 2
    for i, (text, color) in enumerate(steps):
        box(ax, x, y, box_w, box_h, text, color, fontsize=13.5)
        if i < len(steps) - 1:
            arrow(ax, (x + box_w, y + box_h / 2), (x + box_w + gap, y + box_h / 2))
        x += box_w + gap

    ax.text(50, 35.8, "For each functional category, mask its mRNA pixel region, retrain FullSizeCNN, and compare performance with the unmasked baseline.", ha="center", va="center", fontsize=10.8, color="#33465D")

    # Example panels
    cmap = ListedColormap(CATEGORY_COLORS)
    ax0 = fig.add_axes([0.08, 0.03, 0.205, 0.451])
    ax0.imshow(original, cmap="gray", aspect="auto")
    ax0.set_title("A. Before masking", fontsize=12, fontweight="bold")
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_axes([0.315, 0.03, 0.205, 0.451])
    ax1.imshow(cat_grid, cmap=cmap, vmin=0, vmax=5, aspect="auto")
    ax1.set_title("B. Functional category map", fontsize=12, fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])
    center = (cat_grid.shape[0] / 2 - 0.5, cat_grid.shape[1] / 2 - 0.5)
    ax1.scatter(*center, s=80, marker="o", facecolor="none", edgecolor="white", linewidths=2)
    ax1.annotate(
        "center\n(high JSD)",
        xy=center,
        xytext=(center[0] + 1.6, center[1] - 2.2),
        color="white",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="-", color="white", lw=1.0),
    )

    ax2 = fig.add_axes([0.55, 0.03, 0.205, 0.451])
    ax2.imshow(masked, cmap="gray", aspect="auto")
    ax2.set_title("C. After masking \"Other\"", fontsize=12, fontweight="bold")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Color legend for panel B.
    labels = [
        "Development /\nEpithelium",
        "Signaling /\nTransport",
        "Hormone /\nMetabolism",
        "Immune /\nInflammation",
        "Cell cycle /\nProliferation",
        "Other",
    ]
    ax_leg = fig.add_axes([0.82, 0.03, 0.13, 0.451])
    ax_leg.set_xlim(0, 10)
    ax_leg.set_ylim(-0.2, len(labels) + 0.2)
    ax_leg.axis("off")
    ax_leg.set_title("Color\nlegend", fontsize=11, fontweight="bold", loc="left")
    for i, (label, color) in enumerate(zip(labels, CATEGORY_COLORS)):
        y = len(labels) - i - 0.5
        ax_leg.add_patch(plt.Rectangle((0.2, y - 0.30), 1.2, 0.60, color=color, clip_on=False))
        ax_leg.text(1.8, y, label, va="center", ha="left", fontsize=9.8, fontweight="bold", color="#1E3A5F")

    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
