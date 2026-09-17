#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""English technical-route figure v20: larger modules and better spacing."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge


ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = ROOT / "data/paper_figures/fig1_route_v20.png"
OUT_SVG = ROOT / "data/paper_figures/fig1_route_v20.svg"


def rounded_box(ax, x, y, w, h, fc, ec, lw=1.2, zorder=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.10",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=zorder,
        )
    )


def arrow(ax, x0, y0, x1, y1, color="#6F8DA3", lw=2.2):
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


def draw_icon(ax, kind, x, y, color):
    lw = 2.0
    if kind == "db":
        ax.add_patch(Rectangle((x - 0.29, y - 0.03), 0.58, 0.13, color="#BBD5EA", ec=color, lw=lw, zorder=4))
        ax.add_patch(Rectangle((x - 0.29, y - 0.19), 0.58, 0.13, color="#D9EAF5", ec=color, lw=lw, zorder=4))
        ax.add_patch(Rectangle((x - 0.29, y - 0.35), 0.58, 0.13, color="#EAF4F8", ec=color, lw=lw, zorder=4))
    elif kind == "venn":
        ax.add_patch(Circle((x - 0.10, y), 0.22, fill=False, ec=color, lw=lw, zorder=4))
        ax.add_patch(Circle((x + 0.10, y), 0.22, fill=False, ec=color, lw=lw, zorder=4))
    elif kind == "label":
        colors = ["#6AA4C9", "#F2A45A", "#D86C6C", "#81BBC8"]
        for i, c in enumerate(colors):
            ax.add_patch(Rectangle((x - 0.27 + (i % 2) * 0.25, y + 0.02 - (i // 2) * 0.27), 0.20, 0.20, color=c, ec="white", lw=0.8, zorder=4))
    elif kind == "bars":
        heights = [0.15, 0.30, 0.43, 0.58]
        for i, h in enumerate(heights):
            ax.add_patch(Rectangle((x - 0.29 + i * 0.17, y - 0.29), 0.10, h, color="#BDD7EE", ec=color, lw=0.9, zorder=4))
    elif kind == "hist":
        for i in range(4):
            ax.add_patch(Rectangle((x - 0.27 + i * 0.16, y - 0.25), 0.12, 0.20 + (i % 2) * 0.22, color="#BDD7EE", ec=color, lw=0.9, zorder=4))
    elif kind == "spiral":
        import math
        t = [i / 100 * 3.1416 * 2 * 2.4 for i in range(120)]
        r = [0.04 + 0.32 * (i / 120) for i in range(120)]
        ax.plot([x + r[i] * math.cos(t[i]) for i in range(120)], [y + r[i] * math.sin(t[i]) for i in range(120)], color=color, lw=2.0, zorder=4)
    elif kind == "net":
        for cx, cy in [(-0.20, 0.17), (-0.20, -0.17), (0.17, 0.17), (0.17, -0.17)]:
            ax.add_patch(Circle((x + cx, y + cy), 0.09, color="#F9D8AE", ec=color, lw=1.0, zorder=4))
        ax.plot([x - 0.20, x + 0.17], [y + 0.17, y - 0.17], color=color, lw=1.5, zorder=4)
        ax.plot([x - 0.20, x + 0.17], [y - 0.17, y + 0.17], color=color, lw=1.5, zorder=4)
    elif kind == "merge":
        ax.plot([x - 0.33, x - 0.05], [y + 0.22, y + 0.05], color=color, lw=1.8, zorder=4)
        ax.plot([x - 0.33, x - 0.05], [y - 0.22, y - 0.05], color=color, lw=1.8, zorder=4)
        ax.plot([x - 0.05, x + 0.33], [y, y], color=color, lw=1.8, zorder=4)
        ax.add_patch(Circle((x - 0.05, y), 0.17, fill=False, ec=color, lw=1.6, zorder=4))
    elif kind == "gauge":
        ax.add_patch(Wedge((x, y - 0.12), 0.30, 180, 360, width=0.07, facecolor="none", edgecolor=color, lw=1.4, zorder=4))
        ax.plot([x, x + 0.20], [y - 0.12, y + 0.12], color="#D86C6C", lw=2.0, zorder=4)
    else:
        ax.add_patch(Circle((x, y), 0.19, fill=False, ec=color, lw=1.7, zorder=4))


def main():
    fig, ax = plt.subplots(figsize=(16, 7.4), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FBFD")
    ax.set_facecolor("#F8FBFD")

    phases = [
        (
            "I",
            "Data Preparation",
            "#2E6FA6",
            [
                ("db", "Data acquisition", "TCGA-BRCA from GDC / Xena"),
                ("venn", "Sample alignment", "mRNA  |  CNV  |  miRNA"),
                ("label", "Label definition", "PAM50 and survival"),
            ],
            "#EAF4F8",
            "#BDD7EE",
        ),
        (
            "II",
            "Feature Engineering\nand NSRE Imaging",
            "#5E9CC4",
            [
                ("bars", "Feature selection", "Variance / F / L1"),
                ("hist", "NSRE ranking", "Discriminative importance"),
                ("spiral", "Spiral imaging", "Grayscale and category maps"),
            ],
            "#EAF4F8",
            "#C2DFF0",
        ),
        (
            "III",
            "Modeling, Fusion\nand Evaluation",
            "#C75B5B",
            [
                ("net", "Single-omics models", "ML  |  MLP  |  FullSizeCNN"),
                ("merge", "Multi-omics fusion", "Pairwise / triple fusion"),
                ("gauge", "Evaluation", "5-fold CV  |  AUC / C-index"),
            ],
            "#FDF1E2",
            "#F0C7C7",
        ),
    ]

    box_x = [0.45, 5.55, 10.65]
    box_w = 4.70
    box_y = 1.40
    box_h = 5.60
    header_h = 0.84

    for idx, (no, title, color, modules, face, edge) in enumerate(phases):
        x = box_x[idx]
        rounded_box(ax, x, box_y, box_w, box_h, "#FFFFFF", "#C9D9E8", lw=1.2, zorder=2)
        rounded_box(ax, x + 0.08, box_y + box_h - header_h - 0.08, box_w - 0.16, header_h, color, color, lw=0, zorder=3)
        ax.text(x + 0.40, box_y + box_h - header_h / 2 - 0.08, no, ha="center", va="center", fontsize=13, fontweight="bold", color="white", zorder=4)
        ax.text(x + 0.70, box_y + box_h - header_h / 2 - 0.08, title.replace("\n", " "), ha="left", va="center", fontsize=9.5, fontweight="bold", color="white", zorder=4)

        card_w = box_w - 0.20
        card_h = 1.56
        gap = 0.18
        content_top = box_y + box_h - header_h - 0.28
        for j, (icon_kind, item_title, item_desc) in enumerate(modules):
            card_y = content_top - (j + 1) * card_h - j * gap
            rounded_box(ax, x + 0.10, card_y, card_w, card_h, face, edge, lw=0.9, zorder=3)
            icon_x = x + 0.66
            icon_y = card_y + card_h / 2
            draw_icon(ax, icon_kind, icon_x, icon_y, color)
            ax.text(x + 1.12, card_y + card_h - 0.46, item_title, ha="left", va="center", fontsize=9.4, fontweight="bold", color=color, zorder=4)
            ax.text(x + 1.12, card_y + 0.42, item_desc, ha="left", va="center", fontsize=8.2, color="#4B5B6B", zorder=4)

    arrow(ax, box_x[0] + box_w + 0.08, box_y + box_h / 2, box_x[1] - 0.10, box_y + box_h / 2)
    arrow(ax, box_x[1] + box_w + 0.08, box_y + box_h / 2, box_x[2] - 0.10, box_y + box_h / 2)

    band_y = 0.34
    band_h = 0.70
    rounded_box(ax, 0.75, band_y, 14.50, band_h, "#FBF0F0", "#D9A3A3", lw=1.2, zorder=2)
    ax.text(2.55, band_y + band_h / 2, "Interpretability", ha="center", va="center", fontsize=10.0, fontweight="bold", color="#9F3F3F", zorder=4)
    pills = [(3.55, "SHAP"), (6.55, "Grad-CAM / Saliency"), (9.85, "Functional masking")]
    for x, label in pills:
        rounded_box(ax, x, band_y + 0.12, 2.45, 0.46, "#FFFFFF", "#D9A3A3", lw=1.0, zorder=3)
        ax.text(x + 1.225, band_y + band_h / 2, label, ha="center", va="center", fontsize=8.6, fontweight="bold", color="#A23E3E", zorder=4)

    for x in [2.80, 7.90, 13.00]:
        arrow(ax, x, box_y - 0.05, x, band_y + band_h + 0.04, color="#8FA9BE", lw=2.0)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
