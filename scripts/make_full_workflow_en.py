#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete English workflow schematic."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import textwrap


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/paper_figures/fig_full_workflow_en.png"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text, color, text_color="white", fontsize=10.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.12,rounding_size=0.7",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=text_color)


def arrow(ax, p0, p1, color="#5E6E83"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2,
            color=color,
        )
    )


def submodule(ax, x, y, w, h, title, desc, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.5", linewidth=1.0, edgecolor="#C9D9E8", facecolor="white"))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1E3A5F")
    wrapped = textwrap.fill(desc, 20)
    ax.text(x + w / 2, y + 1.3, wrapped, ha="center", va="center", fontsize=7.8, color="#5E6E83")


def main():
    fig = plt.figure(figsize=(15, 8.2), dpi=300)
    fig.patch.set_facecolor("#F7FAFC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 84)
    ax.axis("off")

    ax.text(75, 79, "NSRE-Guided Omics Imaging and Multi-Omics Fusion", ha="center", va="center", fontsize=18, fontweight="bold", color="#173A5E")
    ax.text(75, 75, "A complete workflow for breast cancer molecular subtyping and survival prediction", ha="center", va="center", fontsize=11.5, color="#5E6E83")

    phases = [
        {
            "x": 2,
            "no": "I",
            "title": "Data Preparation",
            "color": "#2E6FA6",
            "subs": [
                ("Data acquisition", "TCGA-BRCA\nmRNA / CNV / miRNA"),
                ("Sample alignment", "Common sample set"),
                ("Label definition", "PAM50 / survival"),
            ],
        },
        {
            "x": 50,
            "no": "II",
            "title": "Feature Engineering & Imaging",
            "color": "#60A5C7",
            "subs": [
                ("Feature selection", "Low-variance / F / L1\nNSRE"),
                ("NSRE ranking", "Class discrimination"),
                ("Scheme-2 imaging", "Center-high importance\nspiral filling"),
            ],
        },
        {
            "x": 98,
            "no": "III",
            "title": "Modeling, Fusion & Evaluation",
            "color": "#D86C6C",
            "subs": [
                ("Single-omics modeling", "Classical ML / MLP\nFullSizeCNN"),
                ("Multi-omics fusion", "Concat / attention\nStacking"),
                ("Cross-validation", "5-fold CV\nAUC / C-index"),
            ],
        },
    ]

    y_top = 55
    col_w = 42
    header_h = 6
    panel_h = 30

    for phase in phases:
        x = phase["x"]
        # Background band
        ax.add_patch(FancyBboxPatch((x - 1, y_top - 1), col_w + 2, header_h + panel_h + 4, boxstyle="round,pad=0.12,rounding_size=1.2", linewidth=1.0, edgecolor="#DDE7F0", facecolor="#F2F6FA"))
        # Header
        box(ax, x, y_top, col_w, header_h, phase["title"], phase["color"], fontsize=11.5)
        # Panel
        panel_x = x + 1
        panel_y = y_top - panel_h - 1
        panel_w = col_w - 2
        ax.add_patch(FancyBboxPatch((panel_x, panel_y), panel_w, panel_h, boxstyle="round,pad=0.08,rounding_size=0.6", linewidth=1.0, edgecolor="#C9D9E8", facecolor="white"))
        # Three horizontal submodules
        sub_w = (panel_w - 2.2) / 3
        sub_h = panel_h - 1.0
        for j, (title, desc) in enumerate(phase["subs"]):
            sx = panel_x + 0.5 + j * (sub_w + 0.6)
            sy = panel_y + 0.5
            submodule(ax, sx, sy, sub_w, sub_h, title, desc, phase["color"])

    # Phase arrows
    arrow(ax, (2 + col_w, y_top + header_h / 2), (50, y_top + header_h / 2))
    arrow(ax, (50 + col_w, y_top + header_h / 2), (98, y_top + header_h / 2))

    # Bottom interpretability band
    ax.text(75, 17, "Downstream Interpretability", ha="center", va="center", fontsize=12, fontweight="bold", color="#6E7A8A")
    items = ["SHAP", "Grad-CAM", "Category masking", "Key factors"]
    item_w = 18
    item_h = 6
    x = 21
    y = 7
    for i, label in enumerate(items):
        box(ax, x + i * (item_w + 3), y, item_w, item_h, label, "#A23E3E", fontsize=9)

    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
