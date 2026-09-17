#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw Figure 7 using Figure-3-style axes and categorical colors."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import spiral_order  # noqa: E402


DATA = ROOT / "data"
IMG = DATA / "images/PAM50/mRNA"
INTERP = DATA / "interpretability"
OUT = DATA / "paper_figures" / "fig7_nsre_v3.png"

CAT_NAMES = {
    0: "Development / Epithelium",
    1: "Signaling / Transport",
    2: "Hormone / Metabolism",
    3: "Immune / Inflammation",
    4: "Cell cycle / Proliferation",
    5: "Other",
}
CAT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12.5,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


def fill_grid(values_by_pos, size=20):
    grid = np.zeros((size, size), dtype=float)
    for pos, val in values_by_pos.items():
        grid[pos] = val
    return grid


def add_coordinate_axes(ax):
    ticks = [0, 5, 10, 15, 20]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis="both", labelsize=9.5, length=2.5, width=0.8, colors="#333333")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def main():
    df = pd.read_csv(DATA / "final_datasets/PAM50/mRNA_PAM50_final.tsv", sep="\t")
    sample = df.iloc[0, 1:].to_numpy(dtype=float)
    order = pd.read_csv(IMG / "order.tsv", sep="\t")
    positions = spiral_order(20)
    pos_gene = {pos: gene for pos, gene in zip(positions[: len(order)], order["feature"].tolist())}
    gene_nsre = order.set_index("feature")["nsre"].to_dict()

    expr = {pos: float(sample[df.columns.get_loc(pos_gene[pos]) - 1]) for pos in pos_gene}
    nsre = {pos: float(gene_nsre[pos_gene[pos]]) for pos in pos_gene}
    shap = pd.read_csv(INTERP / "mrna_pam50_shap_importance.tsv", sep="\t").set_index("gene")["mean_importance"].to_dict()
    cnn = pd.read_csv(INTERP / "mrna_pam50_cnn_saliency_importance.tsv", sep="\t").set_index("gene")["mean_importance"].to_dict()
    shap_map = {pos: float(shap.get(pos_gene[pos], 0.0)) for pos in pos_gene}
    cnn_map = {pos: float(cnn.get(pos_gene[pos], 0.0)) for pos in pos_gene}

    gray = fill_grid(expr)
    nsre_grid = fill_grid(nsre)
    shap_grid = fill_grid(shap_map)
    cnn_grid = fill_grid(cnn_map)
    cat_grid = np.load(DATA / "images/examples/mRNA_category_grid.npy")

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    vmin, vmax = gray.min(), gray.max()
    im0 = axes[0, 0].imshow(gray, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal", extent=(0, 20, 20, 0))
    axes[0, 0].set_title("A  Grayscale expression", loc="left", fontsize=13.5, fontweight="bold", color="#1F4D78")

    im1 = axes[0, 1].imshow(nsre_grid, cmap="magma", aspect="equal", extent=(0, 20, 20, 0))
    axes[0, 1].set_title("B  NSRE importance map", loc="left", fontsize=13.5, fontweight="bold", color="#1F4D78")

    cmap = ListedColormap(CAT_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 6, 1), len(CAT_COLORS))
    im_cat = axes[0, 2].imshow(cat_grid, cmap=cmap, norm=norm, aspect="equal", extent=(0, 20, 20, 0))
    axes[0, 2].set_title("C  Gene functional category", loc="left", fontsize=13.5, fontweight="bold", color="#1F4D78")

    im3 = axes[1, 0].imshow(shap_grid, cmap="viridis", aspect="equal", extent=(0, 20, 20, 0))
    axes[1, 0].set_title("D  SHAP feature importance map", loc="left", fontsize=13.5, fontweight="bold", color="#1F4D78")

    im4 = axes[1, 1].imshow(cnn_grid, cmap="cividis", aspect="equal", extent=(0, 20, 20, 0))
    axes[1, 1].set_title("E  CNN saliency feature map", loc="left", fontsize=13.5, fontweight="bold", color="#1F4D78")

    ax_overlay = axes[1, 2]
    ax_overlay.imshow(gray, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal", extent=(0, 20, 20, 0))
    ax_overlay.imshow(cnn_grid, cmap="hot", alpha=0.65, aspect="equal", extent=(0, 20, 20, 0))
    ax_overlay.set_title("F  Overlay: expression + CNN saliency", loc="left", fontsize=13.5, fontweight="bold", color="#1F4D78")

    for ax in axes.ravel():
        add_coordinate_axes(ax)

    for im, ax in [(im0, axes[0, 0]), (im1, axes[0, 1]), (im_cat, axes[0, 2]), (im3, axes[1, 0]), (im4, axes[1, 1])]:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.10)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=9.5)
        if im is im_cat:
            cbar.set_ticks([i + 0.5 for i in range(6)])
            cbar.set_ticklabels([CAT_NAMES[i].split("/")[0] for i in range(6)])

    fig.subplots_adjust(left=0.035, right=0.90, top=0.94, bottom=0.06, wspace=0.42, hspace=0.26)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
