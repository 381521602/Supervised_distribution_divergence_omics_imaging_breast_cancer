#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean, low-density English technical-route figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = ROOT / "data/paper_figures/fig1_route_v18.png"
OUT_SVG = ROOT / "data/paper_figures/fig1_route_v18.svg"


def rounded_box(ax, x, y, w, h, fc, ec, lw=1.2, zorder=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.12",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=zorder,
        )
    )


def arrow(ax, x0, y0, x1, y1, color="#6F8DA3", lw=2.4):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=lw,
            color=color,
            zorder=6,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(15, 7.8), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.8)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FBFD")
    ax.set_facecolor("#F8FBFD")

    ax.text(
        7.5,
        7.35,
        "JSD-Guided Omics Imaging and Multi-Omics Fusion for Breast Cancer",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#173A5E",
    )
    ax.text(
        7.5,
        6.95,
        "Technical route for molecular subtyping and overall survival prediction",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#5E6E83",
    )

    # Phase data
    phases = [
        (
            "I",
            "DATA PREPARATION",
            "#2E6FA6",
            [
                ("Data acquisition", "TCGA-BRCA from GDC and UCSC Xena"),
                ("Sample alignment", "mRNA  |  CNV  |  miRNA"),
                ("Label definition", "PAM50 four classes  |  OS event and time"),
            ],
        ),
        (
            "II",
            "FEATURE ENGINEERING\nAND JSD IMAGING",
            "#5E9CC4",
            [
                ("Feature selection", "Low-variance  |  F  |  L1"),
                ("JSD ranking", "Class-discriminative importance"),
                ("Spiral imaging", "Grayscale and functional-category maps"),
            ],
        ),
        (
            "III",
            "MODELING, FUSION\nAND EVALUATION",
            "#C75B5B",
            [
                ("Single-omics modeling", "Classical ML  |  MLP  |  FullSizeCNN"),
                ("Multi-omics fusion", "Pairwise and triple fusion"),
                ("Evaluation", "5-fold CV  |  AUC  |  C-index"),
            ],
        ),
    ]

    box_x = [0.55, 5.25, 9.95]
    box_w = 4.15
    box_y = 2.65
    box_h = 3.95
    header_h = 0.82

    for idx, (no, title, color, items) in enumerate(phases):
        x = box_x[idx]
        # Main phase box
        rounded_box(ax, x, box_y, box_w, box_h, "#FFFFFF", "#C9D9E8", lw=1.2, zorder=2)
        # Header
        rounded_box(ax, x + 0.10, box_y + box_h - header_h - 0.08, box_w - 0.20, header_h, color, color, lw=0, zorder=3)
        ax.text(
            x + 0.42,
            box_y + box_h - header_h / 2 - 0.08,
            no,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
            zorder=4,
        )
        ax.text(
            x + 0.78,
            box_y + box_h - header_h / 2 - 0.08,
            title.replace("\n", " "),
            ha="left",
            va="center",
            fontsize=9.8,
            fontweight="bold",
            color="white",
            zorder=4,
            linespacing=1.05,
        )

        content_top = box_y + box_h - header_h - 0.30
        item_h = 1.00
        for j, (item_title, item_desc) in enumerate(items):
            center_y = content_top - j * item_h - item_h / 2
            ax.text(
                x + box_w / 2,
                center_y + 0.13,
                item_title,
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color=color,
                zorder=4,
            )
            ax.text(
                x + box_w / 2,
                center_y - 0.19,
                item_desc,
                ha="center",
                va="center",
                fontsize=8.1,
                color="#4B5B6B",
                zorder=4,
            )

    # Phase arrows
    arrow(ax, box_x[0] + box_w + 0.10, box_y + box_h / 2, box_x[1] - 0.12, box_y + box_h / 2)
    arrow(ax, box_x[1] + box_w + 0.10, box_y + box_h / 2, box_x[2] - 0.12, box_y + box_h / 2)

    # Interpretability band
    band_y = 1.05
    band_h = 1.00
    rounded_box(ax, 0.70, band_y, 13.60, band_h, "#FBF0F0", "#D9A3A3", lw=1.3, zorder=2)
    ax.text(2.50, band_y + 0.50, "INTERPRETABILITY", ha="center", va="center", fontsize=11, fontweight="bold", color="#9F3F3F", zorder=4)

    pills = [
        (3.35, "SHAP"),
        (6.20, "Grad-CAM / Saliency"),
        (9.30, "Functional masking"),
    ]
    for x, label in pills:
        rounded_box(ax, x, band_y + 0.17, 2.35, 0.66, "#FFFFFF", "#D9A3A3", lw=1.1, zorder=3)
        ax.text(x + 1.175, band_y + 0.50, label, ha="center", va="center", fontsize=9.0, fontweight="bold", color="#A23E3E", zorder=4)

    # Down arrows from phase bottoms to interpretability
    for x in [2.625, 7.325, 12.025]:
        arrow(ax, x, box_y - 0.05, x, band_y + band_h + 0.06, color="#8FA9BE", lw=2.0)

    ax.text(
        7.5,
        0.42,
        "Feature selection, scaling, and model fitting are performed within each training fold.",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#6E7A8A",
        style="italic",
    )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
