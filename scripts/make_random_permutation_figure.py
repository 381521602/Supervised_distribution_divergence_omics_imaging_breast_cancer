#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure and statistics for the random-permutation ordering control."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_FIG = DATA / "paper_figures" / "fig_random_permutation.png"
OUT_STATS = DATA / "random_permutation_control_stats.tsv"

BLUE = "#2E74B5"
DARK = "#1F4D78"
TEAL = "#0E7C86"
ORANGE = "#D97706"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


def main():
    null = pd.read_csv(DATA / "random_permutation_control_results.tsv", sep="\t")
    # NSRE reference values (deterministic spiral)
    nsre_ref = {
        ("PAM50", "accuracy"): 0.9316,
        ("PAM50", "macro_f1"): 0.9205,
        ("Survival", "roc_auc"): 0.6768,
        ("Survival", "c_index"): 0.7125,
    }
    panels = [
        ("PAM50", "accuracy", "A. PAM50 Accuracy", "Accuracy", (0.86, 0.97), BLUE),
        ("PAM50", "macro_f1", "B. PAM50 Macro-F1", "Macro-F1", (0.86, 0.97), TEAL),
        ("Survival", "roc_auc", "C. Survival ROC AUC", "ROC AUC", (0.58, 0.75), ORANGE),
        ("Survival", "c_index", "D. Survival C-index", "C-index", (0.62, 0.76), DARK),
    ]

    stats_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    for ax, (task, metric, title, ylabel, ylim, color) in zip(axes.ravel(), panels):
        rnd = null[(null["task"] == task) & (null["ordering"] == "random") & (null["metric"] == metric)]["mean"].values
        expr = null[(null["task"] == task) & (null["ordering"] == "mean_expression") & (null["metric"] == metric)]["mean"].values[0]
        nsre = nsre_ref[(task, metric)]
        # empirical two-sided exceedance p-value: fraction of random runs >= NSRE
        p_val = float(np.mean(rnd >= nsre))
        stats_rows.append({
            "task": task,
            "metric": metric,
            "nsre_spiral": round(nsre, 4),
            "mean_expression": round(expr, 4),
            "random_mean": round(float(np.mean(rnd)), 4),
            "random_std": round(float(np.std(rnd)), 4),
            "random_min": round(float(np.min(rnd)), 4),
            "random_max": round(float(np.max(rnd)), 4),
            "p_random_geq_nsre": round(p_val, 4),
        })

        ax.hist(rnd, bins=10, color=color, alpha=0.55, edgecolor="white")
        ax.axvline(nsre, color=DARK, lw=2.0, ls="-", label=f"NSRE spiral ({nsre:.4f})")
        ax.axvline(expr, color="#B91C1C", lw=1.8, ls="--", label=f"Mean expression ({expr:.4f})")
        ax.set_xlim(*ylim)
        ax.set_xlabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_ylabel("Count (20 random orders)", fontsize=10, fontweight="bold")
        ax.set_title(title, loc="left", fontweight="bold", color=DARK, fontsize=13)
        ax.grid(axis="y", linestyle="--", alpha=0.2, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.legend(fontsize=9.5, frameon=False, loc="upper left")

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300)
    plt.close(fig)

    with OUT_STATS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(stats_rows)

    print(OUT_FIG)
    print(OUT_STATS)
    print(pd.DataFrame(stats_rows).to_string(index=False))


if __name__ == "__main__":
    main()
