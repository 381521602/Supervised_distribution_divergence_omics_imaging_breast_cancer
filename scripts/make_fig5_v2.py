#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw Figure 5 with larger, bold text."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "paper_figures" / "fig4_triple_integration_v2.png"

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
    df = pd.read_csv(DATA / "comprehensive_triple_integration_results.tsv", sep="\t")
    schemes = [
        "Concat_LogisticRegression",
        "Concat_RandomForest",
        "Concat_GradientBoosting",
        "Concat_SVC",
        "Concat_KNN",
        "Concat_MLP",
        "CNN_TripleConcat",
        "CNN_TripleGated",
        "CNN_LateAvg",
    ]
    labels = [s.replace("Concat_", "").replace("CNN_Triple", "CNN") for s in schemes]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.6))
    panels = [
        (axes[0, 0], "PAM50", "accuracy", "A. PAM50: Accuracy", "Accuracy", BLUE, (0.55, 1.0)),
        (axes[0, 1], "PAM50", "macro_f1", "B. PAM50: Macro-F1", "Macro-F1", DARK, (0.55, 1.0)),
        (axes[1, 0], "Survival", "roc_auc", "C. Survival: ROC AUC", "ROC AUC", TEAL, (0.45, 0.85)),
        (axes[1, 1], "Survival", "c_index", "D. Survival: C-index", "C-index", ORANGE, (0.45, 0.85)),
    ]
    for ax, task, metric, title, ylabel, color, ylim in panels:
        d = df[(df["task"] == task) & (df["metric"] == metric)]
        means, errs = [], []
        for s in schemes:
            sub = d[d["scheme"] == s]
            means.append(float(sub["mean"].iloc[0]))
            errs.append(float(sub["std"].iloc[0]))
        x = np.arange(len(schemes))
        ax.bar(x, means, yerr=errs, capsize=2.2, color=color, alpha=0.9, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11, fontweight="bold")
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
