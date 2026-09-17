#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheme-2 illustration with wrapped English category legend."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data/images/PAM50/mRNA_reorder/scheme2_function_center"
OUT = ROOT / "data/interpretability/figures/fig_scheme2_illustration_en.png"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

CATEGORY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]
CATEGORY_LABELS = [
    "Development /\nEpithelium",
    "Signaling /\nTransport",
    "Hormone /\nMetabolism",
    "Immune /\nInflammation",
    "Cell cycle /\nProliferation",
    "Other",
]


def main():
    cat_grid = np.load(IMG / "category_grid.npy").astype(int)
    cmap = ListedColormap(CATEGORY_COLORS)

    fig = plt.figure(figsize=(11, 6.2), dpi=300)
    fig.patch.set_facecolor("#F7FAFC")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.72], wspace=0.06, left=0.05, right=0.97, top=0.90, bottom=0.06)

    ax_img = fig.add_subplot(gs[0, 0])
    im = ax_img.imshow(cat_grid, cmap=cmap, vmin=0, vmax=5, aspect="equal")
    ax_img.set_title("Scheme-2 functional category map\n(center-high importance, center-outward spiral)", fontsize=11.5, fontweight="bold", pad=12)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    # Center marker and spiral direction annotation.
    center = (cat_grid.shape[0] / 2 - 0.5, cat_grid.shape[1] / 2 - 0.5)
    ax_img.scatter(*center, s=110, marker="o", facecolor="none", edgecolor="white", linewidths=2.5)
    ax_img.annotate(
        "center\n(high JSD)",
        xy=center,
        xytext=(center[0] + 2.4, center[1] - 3.5),
        color="white",
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="-", color="white", lw=1.2),
    )

    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.set_xlim(0, 10)
    ax_legend.set_ylim(-0.2, len(CATEGORY_LABELS) + 0.2)
    ax_legend.axis("off")
    ax_legend.set_title("Functional categories", fontsize=11.5, fontweight="bold", pad=12)
    for i, (label, color) in enumerate(zip(CATEGORY_LABELS, CATEGORY_COLORS)):
        y = len(CATEGORY_LABELS) - i - 0.5
        ax_legend.add_patch(plt.Rectangle((0.3, y - 0.32), 1.4, 0.64, color=color, transform=ax_legend.transData, clip_on=False))
        ax_legend.text(2.2, y, label, va="center", ha="left", fontsize=10.5, color="#1E3A5F")

    fig.suptitle("Scheme-2: functional-block grouping with center-outward spiral filling", fontsize=13, fontweight="bold")
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
