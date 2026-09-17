#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""English technical-route figure v21: v19 layout, larger and right-shifted text."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.weight"] = "bold"


ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = ROOT / "data/paper_figures/fig1_route_v25.png"
OUT_SVG = ROOT / "data/paper_figures/fig1_route_v25.svg"


def rounded_box(ax, x, y, w, h, fc, ec, lw=1.2, zorder=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.09",
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


def draw_icon(ax, kind, x, y, color):
    lw = 1.8
    if kind == "db":
        ax.add_patch(Rectangle((x - 0.25, y - 0.05), 0.50, 0.10, color="#BBD5EA", ec=color, lw=lw, zorder=4))
        ax.add_patch(Rectangle((x - 0.25, y - 0.18), 0.50, 0.10, color="#D9EAF5", ec=color, lw=lw, zorder=4))
        ax.add_patch(Rectangle((x - 0.25, y - 0.31), 0.50, 0.10, color="#EAF4F8", ec=color, lw=lw, zorder=4))
    elif kind == "venn":
        ax.add_patch(Circle((x - 0.08, y), 0.18, fill=False, ec=color, lw=lw, zorder=4))
        ax.add_patch(Circle((x + 0.08, y), 0.18, fill=False, ec=color, lw=lw, zorder=4))
    elif kind == "label":
        colors = ["#6AA4C9", "#F2A45A", "#D86C6C", "#81BBC8"]
        for i, c in enumerate(colors):
            ax.add_patch(Rectangle((x - 0.25 + (i % 2) * 0.22, y + 0.02 - (i // 2) * 0.24), 0.18, 0.18, color=c, ec="white", lw=0.6, zorder=4))
    elif kind == "bars":
        heights = [0.12, 0.26, 0.36, 0.50]
        for i, h in enumerate(heights):
            ax.add_patch(Rectangle((x - 0.26 + i * 0.15, y - 0.25), 0.09, h, color="#BDD7EE", ec=color, lw=0.8, zorder=4))
    elif kind == "hist":
        for i in range(4):
            ax.add_patch(Rectangle((x - 0.24 + i * 0.14, y - 0.22), 0.10, 0.16 + (i % 2) * 0.18, color="#BDD7EE", ec=color, lw=0.8, zorder=4))
    elif kind == "spiral":
        import math
        t = [i / 80 * 3.1416 * 2 * 2.2 for i in range(0, 100)]
        r = [0.04 + 0.28 * (i / 100) for i in range(100)]
        ax.plot([x + r[i] * math.cos(t[i]) for i in range(100)], [y + r[i] * math.sin(t[i]) for i in range(100)], color=color, lw=1.6, zorder=4)
    elif kind == "net":
        for cx, cy in [(-0.18, 0.15), (-0.18, -0.15), (0.15, 0.15), (0.15, -0.15)]:
            ax.add_patch(Circle((x + cx, y + cy), 0.07, color="#F9D8AE", ec=color, lw=0.8, zorder=4))
        ax.plot([x - 0.18, x + 0.15], [y + 0.15, y - 0.15], color=color, lw=1.2, zorder=4)
        ax.plot([x - 0.18, x + 0.15], [y - 0.15, y + 0.15], color=color, lw=1.2, zorder=4)
    elif kind == "merge":
        ax.plot([x - 0.28, x - 0.04], [y + 0.18, y + 0.04], color=color, lw=1.5, zorder=4)
        ax.plot([x - 0.28, x - 0.04], [y - 0.18, y - 0.04], color=color, lw=1.5, zorder=4)
        ax.plot([x - 0.04, x + 0.28], [y, y], color=color, lw=1.5, zorder=4)
        ax.add_patch(Circle((x - 0.04, y), 0.14, fill=False, ec=color, lw=1.4, zorder=4))
    elif kind == "gauge":
        ax.add_patch(Wedge((x, y - 0.10), 0.25, 180, 360, width=0.06, facecolor="none", edgecolor=color, lw=1.2, zorder=4))
        ax.plot([x, x + 0.16], [y - 0.10, y + 0.10], color="#D86C6C", lw=1.6, zorder=4)
    else:
        ax.add_patch(Circle((x, y), 0.16, fill=False, ec=color, lw=1.5, zorder=4))


def main():
    fig, ax = plt.subplots(figsize=(14.6, 6.6), dpi=300)
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 6.6)
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
            "Feature Engineering",
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

    box_x = [0.45, 5.15, 9.85]
    box_w = 4.30
    box_y = 1.28
    box_h = 5.15
    header_h = 0.76

    for idx, (no, title, color, modules, face, edge) in enumerate(phases):
        x = box_x[idx]
        rounded_box(ax, x, box_y, box_w, box_h, "#FFFFFF", "#C9D9E8", lw=1.2, zorder=2)
        rounded_box(ax, x + 0.08, box_y + box_h - header_h - 0.07, box_w - 0.16, header_h, color, color, lw=0, zorder=3)
        ax.text(x + 0.36, box_y + box_h - header_h / 2 - 0.07, no, ha="center", va="center", fontsize=15.5, fontweight="bold", color="white", zorder=4)
        ax.text(x + 0.64, box_y + box_h - header_h / 2 - 0.07, title.replace("\n", " "), ha="left", va="center", fontsize=11.6, fontweight="bold", color="white", zorder=4)

        card_w = box_w - 0.18
        card_h = 1.30
        gap = 0.16
        content_top = box_y + box_h - header_h - 0.26
        for j, (icon_kind, item_title, item_desc) in enumerate(modules):
            card_y = content_top - (j + 1) * card_h - j * gap
            rounded_box(ax, x + 0.09, card_y, card_w, card_h, face, edge, lw=0.9, zorder=3)
            icon_x = x + 0.62
            icon_y = card_y + card_h / 2
            draw_icon(ax, icon_kind, icon_x, icon_y, color)
            # Larger text and moved further to the right.
            ax.text(x + 1.18, card_y + card_h - 0.37, item_title, ha="left", va="center", fontsize=12.2, fontweight="bold", color=color, zorder=4)
            ax.text(x + 1.18, card_y + 0.36, item_desc, ha="left", va="center", fontsize=10.4, fontweight="bold", color="#4B5B6B", zorder=4)

    arrow(ax, box_x[0] + box_w + 0.08, box_y + box_h / 2, box_x[1] - 0.10, box_y + box_h / 2)
    arrow(ax, box_x[1] + box_w + 0.08, box_y + box_h / 2, box_x[2] - 0.10, box_y + box_h / 2)

    band_y = 0.32
    band_h = 0.62
    rounded_box(ax, 0.70, band_y, 13.20, band_h, "#FBF0F0", "#D9A3A3", lw=1.2, zorder=2)
    ax.text(2.40, band_y + band_h / 2, "Interpretability", ha="center", va="center", fontsize=11.4, fontweight="bold", color="#9F3F3F", zorder=4)
    pills = [(3.30, "SHAP"), (6.10, "Grad-CAM / Saliency"), (9.10, "Functional masking")]
    for x, label in pills:
        rounded_box(ax, x, band_y + 0.10, 2.20, 0.42, "#FFFFFF", "#D9A3A3", lw=1.0, zorder=3)
        ax.text(x + 1.10, band_y + band_h / 2, label, ha="center", va="center", fontsize=9.8, fontweight="bold", color="#A23E3E", zorder=4)

    for x in [2.60, 7.30, 12.00]:
        arrow(ax, x, box_y - 0.04, x, band_y + band_h + 0.03, color="#8FA9BE", lw=1.8)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
