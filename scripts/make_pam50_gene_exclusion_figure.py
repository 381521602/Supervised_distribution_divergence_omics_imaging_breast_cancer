#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure for the PAM50 50-gene exclusion sensitivity analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "paper_figures" / "fig_pam50_gene_exclusion.png"

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
    df = pd.read_csv(DATA / "pam50_gene_exclusion_results.tsv", sep="\t")
    config_order = ["full_transcriptome", "exclude_pam50_genes", "pam50_only"]
    config_labels = ["Full 400 genes", "Exclude PAM50 genes\n(379 genes)", "PAM50 genes only\n(21 genes)"]
    model_order = ["LogisticRegression", "MLP", "FullSizeCNN"]
    model_labels = ["LogisticRegression", "MLP", "FullSizeCNN"]
    colors = [BLUE, TEAL, ORANGE]

    panels = [
        ("accuracy", "A. PAM50 Accuracy", "Accuracy", (0.6, 1.0)),
        ("macro_f1", "B. PAM50 Macro-F1", "Macro-F1", (0.6, 1.0)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, (metric, title, ylabel, ylim) in zip(axes, panels):
        sub = df[df["metric"] == metric].copy()
        x = np.arange(len(config_order))
        width = 0.26
        for i, (m, mlab, color) in enumerate(zip(model_order, model_labels, colors)):
            vals = []
            errs = []
            for c in config_order:
                row = sub[(sub["config"] == c) & (sub["model"] == m)]
                vals.append(float(row["mean"].iloc[0]))
                errs.append(float(row["std"].iloc[0]))
            offset = (i - 1) * width
            ax.bar(x + offset, vals, yerr=errs, width=width, capsize=2.5,
                   color=color, alpha=0.9, edgecolor="white", linewidth=0.4,
                   label=mlab)
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, loc="left", fontweight="bold", color=DARK, fontsize=13.5)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=11)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.legend(fontsize=10.5, frameon=False, loc="lower right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
