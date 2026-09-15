#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Annotate panel labels on the NSRE interpretability figure."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "interpretability" / "figures" / "nsre_feature_mining_visualization.png"
OUT = ROOT / "data" / "paper_figures" / "fig6_nsre_annotated.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    img = imread(str(SRC))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 200, h / 200), dpi=200)
    ax.imshow(img, interpolation="nearest")
    ax.set_axis_off()
    # 2 rows x 3 columns panel labels; use fractions of image width/height.
    labels = ["A", "B", "C", "D", "E", "F"]
    xs = [0.025, 0.355, 0.685, 0.025, 0.355, 0.685]
    ys = [0.95, 0.95, 0.95, 0.46, 0.46, 0.46]
    for label, x, y in zip(labels, xs, ys):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            color="white",
            ha="left",
            va="top",
            bbox=dict(boxstyle="square,pad=0.12", fc="black", ec="none", alpha=0.72),
        )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT, dpi=200, facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
