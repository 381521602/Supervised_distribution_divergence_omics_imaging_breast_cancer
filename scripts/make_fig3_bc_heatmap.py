# -*- coding: utf-8 -*-
"""Redraw Figure 3B (triple fusion) and 3C (advanced methods) as heatmaps,
then re-merge with the existing 3A single-omics panel."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import imread

D = Path(r"E:\组学数据图像化\data\paper_figures")
DATA = Path(r"E:\组学数据图像化\data")
DARK = "#1F4D78"

A = D / "fig3_single_omics_v3_leakfree.png"
B_OUT = D / "fig3_triple_fusion_heatmap.png"
C_OUT = D / "fig3_advanced_heatmap.png"
OUT = D / "fig3_prediction_performance_leakfree.png"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


def draw_heatmap(mat, row_labels, col_labels, value_fmt, title, out, figsize, na_text="NA"):
    # per-column min-max normalization
    norm = np.full_like(mat, np.nan, dtype=float)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        valid = col[~np.isnan(col)]
        if valid.size:
            lo, hi = float(valid.min()), float(valid.max())
            norm[:, j] = (col - lo) / (hi - lo) if hi > lo else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9.5, fontweight="bold")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10, fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold", color=DARK, fontsize=12)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, na_text, ha="center", va="center", fontsize=10,
                        fontweight="bold", color="#888888")
            else:
                color = "white" if (not np.isnan(norm[i, j]) and norm[i, j] > 0.55) else "#1a1a1a"
                ax.text(j, i, value_fmt.format(v), ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color=color)

    ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.set_label("Per-column normalized", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


def make_B():
    df = pd.read_csv(DATA / "leakfree_cnv_fusion_rerun.tsv", sep="\t")
    schemes = ["Concat_LR", "Concat_MLP", "CNN_LateAvg", "CNN_Gated"]
    labels = ["Concat LR", "Concat MLP", "CNN LateAvg", "CNN Gated"]
    metrics = [("PAM50", "accuracy", "PAM50\nAccuracy"),
               ("Survival", "roc_auc", "Survival\nROC AUC")]
    mat = np.full((len(schemes), len(metrics)), np.nan)
    for i, s in enumerate(schemes):
        for j, (task, metric, _) in enumerate(metrics):
            sub = df[(df["scheme"] == s) & (df["task"] == task) & (df["metric"] == metric)]
            if not sub.empty:
                mat[i, j] = float(sub["mean"].iloc[0])
    draw_heatmap(mat, labels, [m[2] for m in metrics], "{:.3f}",
                 "Triple-omics fusion (leak-free)", B_OUT, figsize=(5.4, 2.9))


def make_C():
    df = pd.read_csv(DATA / "leakfree_advanced_results.tsv", sep="\t")
    methods = ["Transformer", "DeepSurv"]
    metrics = [("PAM50", "accuracy"), ("PAM50", "macro_f1"),
               ("Survival", "roc_auc"), ("Survival", "c_index")]
    labels = ["PAM50\nAccuracy", "PAM50\nMacro-F1", "Survival\nROC AUC", "Survival\nC-index"]
    mat = np.full((len(methods), len(metrics)), np.nan)
    for i, m in enumerate(methods):
        for j, (task, metric) in enumerate(metrics):
            sub = df[(df["method"] == m) & (df["task"] == task) & (df["metric"] == metric)]
            if not sub.empty:
                mat[i, j] = float(sub["mean"].iloc[0])
    draw_heatmap(mat, methods, labels, "{:.3f}",
                 "Advanced methods (leak-free)", C_OUT, figsize=(5.8, 2.9), na_text="NA")


def merge():
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 0.62], hspace=0.06, wspace=0.03,
                          left=0.02, right=0.98, top=0.97, bottom=0.02)
    for ax, path, label in [(fig.add_subplot(gs[0, :]), A, "A"),
                            (fig.add_subplot(gs[1, 0]), B_OUT, "B"),
                            (fig.add_subplot(gs[1, 1]), C_OUT, "C")]:
        img = imread(path)
        ax.imshow(img, interpolation="nearest", aspect="auto")
        ax.set_axis_off()
        ax.text(0.004, 0.985, label, transform=ax.transAxes, fontsize=18, fontweight="bold",
                va="top", ha="left", color="black")
    fig.savefig(OUT, dpi=200, facecolor="white")
    plt.close(fig)
    print("saved merged:", OUT)


def main():
    make_B()
    make_C()
    merge()


if __name__ == "__main__":
    main()
