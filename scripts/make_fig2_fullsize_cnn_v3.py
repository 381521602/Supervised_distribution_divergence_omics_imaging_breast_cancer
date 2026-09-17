#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neat, title-free FullSizeCNN architecture diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.weight"] = "bold"


ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = ROOT / "data/paper_figures/fig2_fullsize_cnn_v4.png"
OUT_SVG = ROOT / "data/paper_figures/fig2_fullsize_cnn_v4.svg"


def box(ax, x, y, w, h, fc, ec, lw=1.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.08",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=2,
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
    fig, ax = plt.subplots(figsize=(13.0, 4.7), dpi=300)
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 4.7)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FBFD")
    ax.set_facecolor("#F8FBFD")

    centers = [1.35, 3.85, 6.35, 8.85, 11.35]
    block_w = 1.75
    block_h = 2.55
    block_y = 1.55
    center_y = block_y + block_h / 2

    # Input image
    x = centers[0] - block_w / 2
    box(ax, x, block_y, block_w, block_h, "#FFFFFF", "#7EA9CC", lw=1.3)
    n = 6
    cell = 0.20
    x0 = centers[0] - n * cell / 2
    y0 = center_y - n * cell / 2
    for i in range(n):
        for j in range(n):
            val = ((i + j) % 3) / 2.0
            ax.add_patch(Rectangle((x0 + i * cell, y0 + j * cell), cell, cell, facecolor=plt.cm.Blues(0.35 + 0.55 * val), edgecolor="white", lw=0.3, zorder=3))

    # Full-size convolution
    x = centers[1] - block_w / 2
    box(ax, x, block_y, block_w, block_h, "#EAF4F8", "#5E9CC4", lw=1.3)
    for k in range(6):
        yy = center_y - 0.75 + k * 0.25
        ax.add_patch(Rectangle((centers[1] - 0.55, yy), 1.10, 0.18, facecolor="#BDD7EE", edgecolor="#5E9CC4", lw=0.6, zorder=3))

    # Feature maps
    x = centers[2] - block_w / 2
    box(ax, x, block_y, block_w, block_h, "#FFFFFF", "#7EA9CC", lw=1.3)
    for k in range(16):
        col = k // 4
        row = k % 4
        xx = centers[2] - 0.54 + col * 0.28
        yy = center_y - 0.42 + row * 0.28
        ax.add_patch(Rectangle((xx, yy), 0.19, 0.19, facecolor="#DCEAF5", edgecolor="#5E9CC4", lw=0.5, zorder=3))

    # Flatten
    x = centers[3] - block_w / 2
    box(ax, x, block_y, block_w, block_h, "#FFFFFF", "#9AAFC0", lw=1.3)
    for k in range(16):
        yy = center_y - 0.80 + k * 0.105
        ax.add_patch(Circle((centers[3], yy), 0.032, color="#9AAFC0", zorder=4))

    # MLP head
    x = centers[4] - block_w / 2
    box(ax, x, block_y, block_w, block_h, "#FDF1E2", "#D9A85E", lw=1.3)
    for k in range(7):
        yy = center_y - 0.72 + k * 0.24
        ax.add_patch(Circle((centers[4] - 0.28, yy), 0.065, color="#F9D8AE", ec="#C98A35", lw=0.7, zorder=4))
    for k in range(4):
        yy = center_y - 0.45 + k * 0.30
        ax.add_patch(Circle((centers[4] + 0.32, yy), 0.065, color="#F7C98A", ec="#B96F2A", lw=0.7, zorder=4))

    # Labels below blocks
    labels = [
        "Input\nH × W × 1",
        "Full-size Conv\nkernel = H × W\n32 filters",
        "Feature maps\n1 × 1 × 32",
        "Flatten\n32",
        "MLP head\n64 → ReLU\n→ Output",
    ]
    for cx, label in zip(centers, labels):
        ax.text(cx, 0.88, label, ha="center", va="center", fontsize=9.8, fontweight="bold", color="#4B5B6B", linespacing=1.05, zorder=4)

    for i in range(4):
        arrow(ax, centers[i] + block_w / 2 + 0.06, center_y, centers[i + 1] - block_w / 2 - 0.06, center_y)

    ax.text(
        6.5,
        0.20,
        "PAM50: H = W = 20, output = 4        Survival: H = W = 15, output = 1",
        ha="center",
        va="center",
        fontsize=9.8,
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
