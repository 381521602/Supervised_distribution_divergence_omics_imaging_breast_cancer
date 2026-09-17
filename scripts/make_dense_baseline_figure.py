#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure comparing Dense-equivalent baseline vs FullSizeCNN."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "paper_figures" / "fig_dense_equivalent.png"

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
    dense = pd.read_csv(DATA / "dense_equivalent_baseline_results.tsv", sep="\t")
    cnn = pd.read_csv(DATA / "comprehensive_batch1_single_omics_results.tsv", sep="\t")
    cnn = cnn[cnn["model"] == "FullSizeCNN"]

    panels = [
        ("PAM50", "accuracy", "A. PAM50 Accuracy", "Accuracy", (0.4, 1.0)),
        ("Survival", "c_index", "B. Survival C-index", "C-index", (0.3, 0.85)),
    ]
    omics_order = ["mRNA", "CNV", "miRNA"]
    colors = [BLUE, TEAL]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    for ax, (task, metric, title, ylabel, ylim) in zip(axes, panels):
        x = np.arange(len(omics_order))
        width = 0.34
        for i, (df, label) in enumerate([(dense, "Dense-equivalent"), (cnn, "FullSizeCNN")]):
            vals, errs = [], []
            for omics in omics_order:
                row = df[(df["omics"] == omics) & (df["task"] == task) & (df["metric"] == metric)]
                vals.append(float(row["mean"].iloc[0]))
                errs.append(float(row["std"].iloc[0]))
            ax.bar(x + (i - 0.5) * width, vals, yerr=errs, width=width, capsize=2.5,
                   color=colors[i], alpha=0.9, edgecolor="white", linewidth=0.4,
                   label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(omics_order, fontsize=12, fontweight="bold")
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
