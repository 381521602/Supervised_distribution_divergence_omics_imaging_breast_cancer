#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure comparing functional-category masks and equal-size random masks."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
FOLDS = ROOT / "data/mrna_equal_mask_control_folds.tsv"
OUT = ROOT / "data/interpretability/figures/fig_mrna_equal_mask_control.png"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    df = pd.read_csv(FOLDS, sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    configs = [
        ("PAM50", "macro_f1", "PAM50 Macro-F1", 0.80),
        ("Survival", "c_index", "Survival C-index", 0.55),
    ]
    for ax, (task, metric, title, ymin) in zip(axes, configs):
        sub = df[(df["task"] == task) & (df["metric"] == metric)].copy()
        sub["kind"] = sub["variant"].map(lambda x: "Category mask" if x.startswith("mask_") else ("Baseline" if x == "baseline" else "Random equal-size"))
        sub["label"] = sub["variant"].map(
            lambda x: "Baseline" if x == "baseline" else x.replace("mask_category_", "C").replace("random_equal_", "R")
        )
        sub["color"] = sub["kind"].map({"Baseline": "#2E6FA6", "Category mask": "#D86C6C", "Random equal-size": "#9AA6B2"})
        for _, row in sub.iterrows():
            ax.bar(row["label"], row["value"], color=row["color"], alpha=0.9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_ylim(ymin, sub["value"].max() + 0.04)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle("mRNA scheme-2: functional-category mask vs equal-size random mask", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
