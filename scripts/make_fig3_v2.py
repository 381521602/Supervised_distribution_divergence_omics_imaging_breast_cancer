#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw Figure 3 using the same categorical color family as Figure 7."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "paper_figures" / "fig3_single_omics_v3.png"

MODEL_COLORS = {
    "LogisticRegression": "#1f77b4",
    "RandomForest": "#ff7f0e",
    "GradientBoosting": "#2ca02c",
    "SVC": "#d62728",
    "KNN": "#9467bd",
    "MLP": "#8c564b",
    "FullSizeCNN": "#17becf",
    "CoxPH": "#1f4d78",
}

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


def read_tsv(name):
    return pd.read_csv(DATA / name, sep="\t")


def grouped_bar(ax, df, metric, groups, models, title, ylabel, ylim):
    x = np.arange(len(groups))
    n = len(models)
    width = 0.78 / n
    for i, model in enumerate(models):
        means, errs = [], []
        for g in groups:
            sub = df[(df["omics"] == g) & (df["model"] == model) & (df["metric"] == metric)]
            if sub.empty:
                means.append(np.nan)
                errs.append(0.0)
            else:
                means.append(float(sub["mean"].iloc[0]))
                errs.append(float(sub["std"].iloc[0]))
        ax.bar(
            x + (i - (n - 1) / 2) * width,
            means,
            width * 0.88,
            yerr=errs,
            capsize=2.2,
            color=MODEL_COLORS.get(model, "#1f77b4"),
            label=model,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold", color="#1F4D78", fontsize=13.5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def main():
    df = read_tsv("comprehensive_batch1_single_omics_results.tsv")
    models_pam = [
        "LogisticRegression",
        "RandomForest",
        "GradientBoosting",
        "SVC",
        "KNN",
        "MLP",
        "FullSizeCNN",
    ]
    models_sur = models_pam + ["CoxPH"]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.2))
    panels = [
        (axes[0, 0], df[df["task"] == "PAM50"], "accuracy", models_pam, "A. PAM50: Accuracy", "Accuracy", (0.45, 1.0)),
        (axes[0, 1], df[df["task"] == "PAM50"], "macro_f1", models_pam, "B. PAM50: Macro-F1", "Macro-F1", (0.40, 1.0)),
        (axes[1, 0], df[df["task"] == "Survival"], "roc_auc", models_sur, "C. Survival: ROC AUC", "ROC AUC", (0.45, 0.85)),
        (axes[1, 1], df[df["task"] == "Survival"], "c_index", models_sur, "D. Survival: C-index", "C-index", (0.45, 0.85)),
    ]
    for ax, sub, metric, models, title, ylabel, ylim in panels:
        grouped_bar(ax, sub, metric, ["mRNA", "CNV", "miRNA"], models, title, ylabel, ylim)

    all_models = list(dict.fromkeys(models_pam + models_sur))
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in all_models]
    fig.legend(
        handles,
        all_models,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        prop={"size": 11, "weight": "bold"},
        handlelength=1.0,
        handleheight=0.7,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
