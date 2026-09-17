#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a publication-ready English technical-route figure."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = ROOT / "data/paper_figures/fig1_technical_route_v17.png"
OUT_SVG = ROOT / "data/paper_figures/fig1_technical_route_v17.svg"


def draw_box(ax, x, y, w, h, facecolor, edgecolor, linewidth=1.0, alpha=1.0, zorder=2):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def draw_module(ax, x, y, w, h, title, body_lines, title_color, body_color, facecolor, edgecolor):
    draw_box(ax, x, y, w, h, facecolor, edgecolor, linewidth=1.0)
    ax.text(
        x + w / 2,
        y + h - 0.22,
        title,
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=title_color,
        zorder=4,
    )
    n = len(body_lines)
    start_y = y + h - 0.52
    for i, line in enumerate(body_lines):
        yy = start_y - i * 0.26
        ax.text(
            x + w / 2,
            yy,
            line,
            ha="center",
            va="center",
            fontsize=7.7,
            color=body_color,
            zorder=4,
            linespacing=1.08,
        )


def arrow(ax, x0, y0, x1, y1, color="#7D9CB7", linewidth=2.2, style="-|>", mutation_scale=16, zorder=5):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            zorder=zorder,
        )
    )


def wrap_lines(text: str, width: int = 34):
    return textwrap.wrap(text, width=width) or [text]


def main():
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FBFD")
    ax.set_facecolor("#F8FBFD")

    # Title block
    ax.text(7.0, 6.88, "JSD-Guided Omics Imaging and Multi-Omics Fusion", ha="center", va="center", fontsize=18, fontweight="bold", color="#173A5E")
    ax.text(7.0, 6.48, "Technical route for breast cancer molecular subtyping and overall survival prediction", ha="center", va="center", fontsize=10.5, color="#5E6E83")

    # Phase definitions: (label, title, color, modules)
    phases = [
        (
            "I",
            "Data Preparation",
            "#2E6FA6",
            [
                ("Data acquisition", ["TCGA-BRCA from GDC", "and UCSC Xena"]),
                ("Sample alignment", ["mRNA / CNV / miRNA", "primary tumor barcodes"]),
                ("Label definition", ["PAM50 four classes", "OS event and OS time"]),
            ],
            "#EAF4F8",
            "#BDD7EE",
        ),
        (
            "II",
            "Feature Engineering\nand JSD Imaging",
            "#5E9CC4",
            [
                ("Feature selection", ["Low-variance / F / L1", "within each training fold"]),
                ("JSD ranking", ["Class-discriminative", "symmetric relative entropy"]),
                ("Center-out spiral imaging", ["Grayscale expression maps", "functional category maps"]),
            ],
            "#EAF4F8",
            "#C2DFF0",
        ),
        (
            "III",
            "Modeling, Fusion\nand Evaluation",
            "#C75B5B",
            [
                ("Single-omics modeling", ["Classical ML / MLP", "FullSizeCNN"]),
                ("Multi-omics fusion", ["Pairwise and triple fusion", "concat / gating / late average"]),
                ("Evaluation", ["5-fold CV", "Accuracy / Macro-F1 / AUC / C-index"]),
            ],
            "#FDF1E2",
            "#F0C7C7",
        ),
    ]

    col_x = [0.55, 5.05, 9.55]
    col_w = 3.95
    header_h = 0.72
    top_y = 5.32
    body_h = 4.05
    body_y = top_y - header_h - 0.05
    module_h = 1.08
    gap = 0.18

    for idx, (no, title, color, modules, face, edge) in enumerate(phases):
        x = col_x[idx]
        # Phase background
        draw_box(ax, x - 0.10, body_y - 0.12, col_w + 0.20, body_h + 0.24, "#F2F6FA", "#DDE7F0", linewidth=1.0, zorder=1)
        # Header
        draw_box(ax, x, top_y, col_w, header_h, color, color, linewidth=0.0, zorder=3)
        ax.text(x + 0.34, top_y + header_h / 2, no, ha="center", va="center", fontsize=13, fontweight="bold", color="white", zorder=4)
        ax.text(x + 0.72, top_y + header_h / 2, title.replace("\n", " "), ha="left", va="center", fontsize=10.8, fontweight="bold", color="white", zorder=4)

        # Modules
        for j, (mod_title, body_lines) in enumerate(modules):
            yy = body_y + body_h - module_h - j * (module_h + gap)
            draw_module(
                ax,
                x + 0.05,
                yy,
                col_w - 0.10,
                module_h,
                mod_title,
                body_lines,
                color,
                "#4B5B6B",
                face,
                edge,
            )

    # Horizontal arrows between phases
    arrow(ax, col_x[0] + col_w + 0.02, top_y + header_h / 2, col_x[1] - 0.05, top_y + header_h / 2)
    arrow(ax, col_x[1] + col_w + 0.02, top_y + header_h / 2, col_x[2] - 0.05, top_y + header_h / 2)

    # Downward arrows to interpretability
    interp_y = 0.58
    for x in [2.525, 7.025, 11.525]:
        arrow(ax, x, body_y - 0.18, x, interp_y + 0.30, color="#8FA9BE", linewidth=1.8)

    # Interpretability band
    draw_box(ax, 0.60, interp_y, 12.80, 0.66, "#FBF0F0", "#D9A3A3", linewidth=1.2, zorder=2)
    ax.text(2.45, interp_y + 0.33, "Interpretability", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#9F3F3F", zorder=4)
    draw_box(ax, 3.45, interp_y + 0.08, 2.30, 0.50, "#FFFFFF", "#D9A3A3", linewidth=1.0, zorder=3)
    ax.text(4.60, interp_y + 0.33, "SHAP", ha="center", va="center", fontsize=10, fontweight="bold", color="#A23E3E", zorder=4)
    draw_box(ax, 6.15, interp_y + 0.08, 2.30, 0.50, "#FFFFFF", "#D9A3A3", linewidth=1.0, zorder=3)
    ax.text(7.30, interp_y + 0.33, "Grad-CAM / Saliency", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#A23E3E", zorder=4)
    draw_box(ax, 8.85, interp_y + 0.08, 2.80, 0.50, "#FFFFFF", "#D9A3A3", linewidth=1.0, zorder=3)
    ax.text(10.25, interp_y + 0.33, "Functional masking", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#A23E3E", zorder=4)
    ax.text(7.0, interp_y - 0.20, "Key gene identification and module-level contribution analysis", ha="center", va="center", fontsize=9.0, color="#6E7A8A", zorder=4)

    ax.text(
        7.0,
        0.16,
        "Feature selection, scaling, and model fitting are performed within each training fold.",
        ha="center",
        va="center",
        fontsize=8.4,
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
