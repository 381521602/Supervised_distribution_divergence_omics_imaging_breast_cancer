#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw a modern FullSizeCNN architecture diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = ROOT / "data/paper_figures/fig2_fullsize_cnn_v2.png"
OUT_SVG = ROOT / "data/paper_figures/fig2_fullsize_cnn_v2.svg"


def box(ax, x, y, w, h, fc, ec, lw=1.2, rounding=0.08, zorder=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.012,rounding_size={rounding}",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=zorder,
        )
    )


def arrow(ax, x0, y0, x1, y1, color="#6F8DA3", lw=2.0):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=lw,
            color=color,
            zorder=6,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(14.5, 6.2), dpi=300)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FBFD")
    ax.set_facecolor("#F8FBFD")

    ax.text(0.55, 5.85, "FullSizeCNN", ha="left", va="center", fontsize=18, fontweight="bold", color="#173A5E")
    ax.text(0.55, 5.35, "A compact convolutional network for NSRE-ordered omics images", ha="left", va="center", fontsize=10.5, color="#5E6E83")

    # 1. Input omics image
    box(ax, 0.55, 1.35, 2.10, 3.15, "#FFFFFF", "#7EA9CC", lw=1.4)
    ax.text(1.60, 1.05, "Input omics image", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2E6FA6")
    n = 7
    grid_x0, grid_y0 = 0.83, 1.68
    cell = 0.21
    for i in range(n):
        for j in range(n):
            val = ((i + j) % 3) / 2.0
            color = plt.cm.Blues(0.35 + 0.55 * val)
            ax.add_patch(Rectangle((grid_x0 + i * cell, grid_y0 + j * cell), cell, cell, facecolor=color, edgecolor="white", lw=0.4, zorder=3))
    ax.text(1.60, 4.65, "H × W × 1", ha="center", va="center", fontsize=8.5, color="#4B5B6B", zorder=4)

    # 2. Full-size convolution cube
    box(ax, 3.55, 1.35, 2.20, 3.15, "#EAF4F8", "#5E9CC4", lw=1.4)
    ax.text(4.65, 1.08, "Full-size convolution", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2E6FA6")
    for k in range(5):
        yy = 1.68 + k * 0.50
        ax.add_patch(Rectangle((3.82, yy), 1.28, 0.40, facecolor="#BDD7EE", edgecolor="#5E9CC4", lw=0.7, zorder=3))
    ax.text(4.65, 4.55, "kernel = H × W\n32 filters", ha="center", va="center", fontsize=8.4, color="#4B5B6B", zorder=4)

    # 3. Feature maps
    box(ax, 6.75, 1.35, 2.10, 3.15, "#FFFFFF", "#7EA9CC", lw=1.4)
    ax.text(7.80, 1.05, "Feature maps", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2E6FA6")
    for k in range(32):
        col = k // 8
        row = k % 8
        xx = 7.02 + col * 0.42
        yy = 1.72 + row * 0.30
        ax.add_patch(Rectangle((xx, yy), 0.25, 0.20, facecolor="#DCEAF5", edgecolor="#5E9CC4", lw=0.5, zorder=3))
    ax.text(7.80, 4.55, "1 × 1 × 32", ha="center", va="center", fontsize=8.5, color="#4B5B6B", zorder=4)

    # 4. Flatten
    box(ax, 10.00, 1.55, 1.25, 2.75, "#FFFFFF", "#9AAFC0", lw=1.3)
    ax.text(10.625, 1.25, "Flatten", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#4B5B6B")
    for k in range(16):
        yy = 1.80 + k * 0.14
        ax.add_patch(Circle((10.625, yy), 0.045, color="#9AAFC0", zorder=4))

    # 5. MLP head
    box(ax, 12.35, 1.35, 1.95, 3.15, "#FDF1E2", "#D9A85E", lw=1.4)
    ax.text(13.325, 1.08, "MLP head", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#A35E00")
    for k in range(8):
        yy = 1.68 + k * 0.31
        ax.add_patch(Circle((12.80, yy), 0.085, color="#F9D8AE", ec="#C98A35", lw=0.8, zorder=4))
    for k in range(4):
        yy = 2.20 + k * 0.58
        ax.add_patch(Circle((13.80, yy), 0.085, color="#F7C98A", ec="#B96F2A", lw=0.8, zorder=4))
    ax.text(13.325, 4.55, "64 → ReLU\n→ Dropout → Output", ha="center", va="center", fontsize=7.8, color="#6E5A3A", zorder=4)

    # Arrows between blocks
    arrow(ax, 2.70, 2.925, 3.50, 2.925)
    arrow(ax, 5.80, 2.925, 6.70, 2.925)
    arrow(ax, 8.90, 2.925, 9.95, 2.925)
    arrow(ax, 11.30, 2.925, 12.30, 2.925)

    ax.text(
        7.25,
        0.42,
        "PAM50: H = W = 20, output = 4        Survival: H = W = 15, output = 1",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color="#5E6E83",
    )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
