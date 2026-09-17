#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw Figure 9 with larger, bold text."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "interpretability" / "figures" / "fig_other_omics_category_masking_v2.png"

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
    omics = ["CNV", "miRNA"]
    tasks = ["PAM50", "Survival"]
    metric_map = {"PAM50": ("accuracy", "Accuracy"), "Survival": ("c_index", "C-index")}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for i, om in enumerate(omics):
        for j, task in enumerate(tasks):
            ax = axes[j, i]
            metric, label = metric_map[task]
            sub = df[(df["omics"] == om) & (df["task"] == task) & (df["metric"] == metric)].copy()
            sub = sub.sort_values("category")
            colors = ["#2E6FA6" if r["category"] == "Baseline" else "#D86C6C" for _, r in sub.iterrows()]
            ax.bar(sub["category"], sub["mean"], yerr=sub["std"], capsize=4, color=colors, alpha=0.9)
            ax.set_title(f"{om} · {task} · {label}", fontsize=13.5, fontweight="bold")
            ax.tick_params(axis="x", rotation=22, labelsize=11)
            ax.tick_params(axis="y", labelsize=11)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.set_ylim(0, max(sub["mean"].max() + sub["std"].max() + 0.05, 0.85))
    fig.suptitle("CNV and miRNA functional-category masking", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
