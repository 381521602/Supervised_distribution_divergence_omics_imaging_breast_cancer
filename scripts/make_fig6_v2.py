#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw Figure 6 with larger, bold text."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "paper_figures" / "fig5_advanced_methods_v2.png"

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
    files = {
        "Multi-task": "advanced_method1_multitask_results.tsv",
        "DeepSurv": "advanced_method2_deepsurv_results.tsv",
        "Transformer": "advanced_method3_transformer_results.tsv",
        "Low-rank bilinear": "advanced_method4_lowrank_bilinear_results.tsv",
        "GNN": "advanced_method5_gnn_results.tsv",
    }
    rows = []
    for method, fname in files.items():
        df = pd.read_csv(DATA / fname, sep="\t")
        for task, metric in [("PAM50", "accuracy"), ("PAM50", "macro_f1"), ("Survival", "roc_auc"), ("Survival", "c_index")]:
            if "task" in df.columns:
                sub = df[(df["task"] == task) & (df["metric"] == metric)]
            else:
                sub = df[(df["metric"] == metric)]
            if not sub.empty:
                rows.append((method, task, metric, float(sub["mean"].iloc[0]), float(sub["std"].iloc[0])))
    d = pd.DataFrame(rows, columns=["method", "task", "metric", "mean", "std"])
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    panels = [
        (axes[0, 0], "PAM50", "accuracy", ["Multi-task", "Transformer", "Low-rank bilinear", "GNN"], "A. PAM50: Accuracy", "Accuracy", BLUE, (0.55, 1.0)),
        (axes[0, 1], "PAM50", "macro_f1", ["Multi-task", "Transformer", "Low-rank bilinear", "GNN"], "B. PAM50: Macro-F1", "Macro-F1", DARK, (0.55, 1.0)),
        (axes[1, 0], "Survival", "roc_auc", ["Multi-task", "DeepSurv", "Transformer", "Low-rank bilinear", "GNN"], "C. Survival: ROC AUC", "ROC AUC", TEAL, (0.45, 0.80)),
        (axes[1, 1], "Survival", "c_index", ["Multi-task", "DeepSurv", "Transformer", "Low-rank bilinear", "GNN"], "D. Survival: C-index", "C-index", ORANGE, (0.45, 0.80)),
    ]
    for ax, task, metric, methods, title, ylabel, color, ylim in panels:
        x = np.arange(len(methods))
        means, errs = [], []
        for m in methods:
            sub = d[(d["method"] == m) & (d["task"] == task) & (d["metric"] == metric)]
            means.append(float(sub["mean"].iloc[0]))
            errs.append(float(sub["std"].iloc[0]))
        ax.bar(x, means, yerr=errs, capsize=2.5, color=color, alpha=0.9, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=11, fontweight="bold")
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
