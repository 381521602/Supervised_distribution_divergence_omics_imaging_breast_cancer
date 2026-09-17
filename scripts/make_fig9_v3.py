#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw Figure 9 in the same style as Figures 4-6."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "interpretability" / "figures" / "fig_other_omics_category_masking_v3.png"

BLUE = "#2E74B5"
DARK = "#1F4D78"
TEAL = "#0E7C86"
ORANGE = "#D97706"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


def main():
    df = pd.read_csv(DATA / "other_omics_category_masking_results.tsv", sep="\t")
    short_names = {
        "Baseline": "Baseline",
        "Development/Epithelium": "Dev",
        "Signaling/Transport": "Signal",
        "Hormone/Metabolism": "Hormone",
        "Immune/Inflammation": "Immune",
        "Cell cycle/Proliferation": "Cell cycle",
        "Other": "Other",
    }

    panels = [
        ("CNV", "PAM50", "accuracy", "A. CNV PAM50: Accuracy", "Accuracy", BLUE, (0.0, 1.0)),
        ("CNV", "Survival", "c_index", "B. CNV Survival: C-index", "C-index", DARK, (0.0, 0.85)),
        ("miRNA", "PAM50", "accuracy", "C. miRNA PAM50: Accuracy", "Accuracy", TEAL, (0.0, 1.0)),
        ("miRNA", "Survival", "c_index", "D. miRNA Survival: C-index", "C-index", ORANGE, (0.0, 0.85)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
    for ax, (omics, task, metric, title, ylabel, color, ylim) in zip(axes.ravel(), panels):
        sub = df[(df["omics"] == omics) & (df["task"] == task) & (df["metric"] == metric)].copy()
        ordered = ["Baseline", "Development/Epithelium", "Signaling/Transport", "Hormone/Metabolism",
                   "Immune/Inflammation", "Cell cycle/Proliferation", "Other"]
        present = [c for c in ordered if c in set(sub["category"])]
        sub = sub.set_index("category").loc[present].reset_index()
        x = np.arange(len(sub))
        ax.bar(x, sub["mean"], yerr=sub["std"], capsize=2.5, color=color, alpha=0.9,
               edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([short_names[c] for c in sub["category"]], rotation=20, ha="right",
                           fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, loc="left", fontweight="bold", color=DARK, fontsize=13.5)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=11)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
