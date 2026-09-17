#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw Figure 3 with Figure-7 categorical color annotation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import spiral_order  # noqa: E402


DATA = ROOT / "data"
PAM_DIR = DATA / "final_datasets/PAM50"
SUR_DIR = DATA / "final_datasets/Survival"
IMG_PAM = DATA / "images/PAM50/mRNA"
IMG_SUR = DATA / "images/Survival/mRNA"
OUT = DATA / "paper_figures/fig3_example_v8.png"

CAT_NAMES = {
    0: "Development / Epithelium",
    1: "Signaling / Transport",
    2: "Hormone / Metabolism",
    3: "Immune / Inflammation",
    4: "Cell cycle / Proliferation",
    5: "Other",
}
CAT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]


def fill_grid(values_by_pos, size):
    grid = np.zeros((size, size), dtype=float)
    for pos, val in values_by_pos.items():
        grid[pos] = val
    return grid


def load_example(omics_dir, final_df, size, sample_idx=0):
    sample = final_df.iloc[sample_idx, 1:].to_numpy(dtype=float)
    order = pd.read_csv(omics_dir / "order.tsv", sep="\t")
    positions = spiral_order(size)
    pos_gene = {pos: gene for pos, gene in zip(positions[: len(order)], order["feature"].tolist())}
    gene_nsre = order.set_index("feature")["nsre"].to_dict()
    expr = {pos: float(sample[final_df.columns.get_loc(pos_gene[pos]) - 1]) for pos in pos_gene}
    nsre = {pos: float(gene_nsre[pos_gene[pos]]) for pos in pos_gene}
    return fill_grid(expr, size), fill_grid(nsre, size)


def add_coordinate_axes(ax, size):
    ticks = [0, 5, 10, 15] if size == 15 else [0, 5, 10, 15, 20]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis="both", labelsize=6.5, length=2.5, width=0.8, colors="#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def main():
    pam_df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
    sur_df = pd.read_csv(SUR_DIR / "mRNA_Survival_final.tsv", sep="\t")
    pam_expr, pam_nsre = load_example(IMG_PAM, pam_df, 20)
    sur_expr, sur_nsre = load_example(IMG_SUR, sur_df, 15)
    pam_cat = np.load(DATA / "images/examples/mRNA_category_grid.npy")
    sur_cat = np.load(DATA / "images/examples/Survival_mRNA_category_grid.npy")

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.8), dpi=300)
    rows = [
        ("A", pam_expr, pam_nsre, pam_cat, 20),
        ("B", sur_expr, sur_nsre, sur_cat, 15),
    ]

    for r, (row_label, expr_grid, nsre_grid, cat_grid, size) in enumerate(rows):
        extent = (0, size, size, 0)
        vmin, vmax = expr_grid.min(), expr_grid.max()
        im_expr = axes[r, 0].imshow(expr_grid, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal", extent=extent)
        axes[r, 0].set_title(f"{row_label}1  Grayscale expression", loc="left", fontsize=9, fontweight="bold", color="#1F4D78")

        im_nsre = axes[r, 1].imshow(nsre_grid, cmap="magma", vmin=0, vmax=float(nsre_grid.max()), aspect="equal", extent=extent)
        axes[r, 1].set_title(f"{row_label}2  NSRE importance map", loc="left", fontsize=9, fontweight="bold", color="#1F4D78")

        cmap = ListedColormap(CAT_COLORS)
        norm = BoundaryNorm(np.arange(-0.5, 6, 1), len(CAT_COLORS))
        im_cat = axes[r, 2].imshow(cat_grid, cmap=cmap, norm=norm, aspect="equal", extent=extent)
        axes[r, 2].set_title(f"{row_label}3  Gene functional category", loc="left", fontsize=9, fontweight="bold", color="#1F4D78")

        for col, im in enumerate([im_expr, im_nsre, im_cat]):
            ax = axes[r, col]
            add_coordinate_axes(ax, size)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=7)
            if col == 2:
                cbar.set_ticks([i + 0.5 for i in range(6)])
                cbar.set_ticklabels([CAT_NAMES[i].split("/")[0] for i in range(6)])

    fig.subplots_adjust(left=0.035, right=0.90, top=0.92, bottom=0.06, wspace=0.35, hspace=0.22)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
