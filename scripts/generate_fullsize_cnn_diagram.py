#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate FullSizeCNN architecture diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/images/report_figures/fullsize_cnn_architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def box(ax, xy, w, h, text, fill="#E8EEF5", fs=10):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.2, edgecolor="#1F4D78", facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#0B2545")


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#2E74B5"))


def main():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Input image
    box(ax, (0.2, 1.9), 1.3, 1.2, "Input\nH × W × 1", fill="#DCE6F1")
    # Full-size conv
    box(ax, (2.2, 1.7), 2.0, 1.6, "Full-size Conv\nkernel = H × W\n32 filters", fill="#BDD7EE")
    # Feature maps
    box(ax, (5.0, 1.7), 1.4, 1.6, "Feature maps\n1 × 1 × 32", fill="#DEEAF6")
    # Flatten
    box(ax, (7.1, 1.7), 1.2, 1.6, "Flatten\n32", fill="#DCE6F1")
    # MLP head
    box(ax, (9.0, 1.5), 1.6, 2.0, "MLP head\n64\nReLU + Dropout\nOutput", fill="#FFF2CC")

    arrow(ax, (1.5, 2.5), (2.2, 2.5))
    arrow(ax, (4.2, 2.5), (5.0, 2.5))
    arrow(ax, (6.4, 2.5), (7.1, 2.5))
    arrow(ax, (8.3, 2.5), (9.0, 2.5))

    ax.text(6.0, 4.45, "FullSizeCNN architecture", ha="center", fontsize=14, fontweight="bold", color="#1F4D78")
    ax.text(6.0, 3.75, "PAM50: H=W=20, output=4; Survival: H=W=15, output=1", ha="center", fontsize=10, color="#333333")

    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved ->", OUT)


if __name__ == "__main__":
    main()
