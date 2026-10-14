#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vertical three-panel masking workflow (B1-B3) for Fig4."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data/images/PAM50/mRNA_reorder/scheme2_function_center"
OUT = ROOT / "data/interpretability/figures/fig_scheme2_masking_flow_en_v5_vertical.png"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

CATEGORY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]
SHORT = ["Dev", "Signal", "Hormone", "Immune", "Cell cycle", "Other"]
mRNA_CMAP = LinearSegmentedColormap.from_list("mRNA", ["#FFFFFF", "#2F6FA3"])


def main():
    gray = np.load(IMG / "images.npy").astype(np.float32)
    cat_grid = np.load(IMG / "category_grid.npy").astype(int)
    original = gray[0, 0]
    masked = original.copy()
    masked[cat_grid == 5] = 0.0

    fig, axes = plt.subplots(3, 1, figsize=(3.6, 9.2), dpi=300)
    im1 = axes[0].imshow(original, cmap=mRNA_CMAP, aspect="equal")
    axes[0].set_title("B1  Before masking", loc="left", fontsize=11.5, fontweight="bold", color="#1F4D78")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="6%", pad=0.06)
    cb0 = fig.colorbar(im1, cax=cax0)
    cb0.set_label("Expression", fontsize=7.5)
    cb0.ax.tick_params(labelsize=6.5)

    cmap = ListedColormap(CATEGORY_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 6, 1), len(CATEGORY_COLORS))
    im = axes[1].imshow(cat_grid, cmap=cmap, norm=norm, aspect="equal")
    axes[1].set_title("B2  Functional category", loc="left", fontsize=11.5, fontweight="bold", color="#1F4D78")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="6%", pad=0.06)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([i + 0.5 for i in range(6)])
    cbar.set_ticklabels(SHORT)
    cbar.ax.tick_params(labelsize=6.5)

    im3 = axes[2].imshow(masked, cmap=mRNA_CMAP, aspect="equal")
    axes[2].set_title('B3  After masking "Other"', loc="left", fontsize=11.5, fontweight="bold", color="#1F4D78")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    divider2 = make_axes_locatable(axes[2])
    cax2 = divider2.append_axes("right", size="6%", pad=0.06)
    cb2 = fig.colorbar(im3, cax=cax2)
    cb2.set_label("Expression", fontsize=7.5)
    cb2.ax.tick_params(labelsize=6.5)

    fig.subplots_adjust(left=0.12, right=0.88, top=0.96, bottom=0.03, hspace=0.28)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
