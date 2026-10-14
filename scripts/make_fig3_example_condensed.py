#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Condensed omics-image example panel for the merged Figure 2 (B1-B6)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import spiral_order  # noqa: E402

DATA = ROOT / "data"
PAM_DIR = DATA / "final_datasets/PAM50"
IMG_PAM = DATA / "images/PAM50"
OUT = DATA / "paper_figures/fig3_example_v12_condensed.png"

CAT_NAMES = {
    0: "Development / Epithelium",
    1: "Signaling / Transport",
    2: "Hormone / Metabolism",
    3: "Immune / Inflammation",
    4: "Cell cycle / Proliferation",
    5: "Other",
}
CAT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]
CAT_COLORS_SOFT = ["#A6C8E4", "#F0B27A", "#A9DFBF", "#F1948A", "#C3A6D9", "#D5D8DC"]
CAT_COLORS_MID = ["#4A90C4", "#F09A4B", "#5CAF7A", "#E06A5E", "#9B72C9", "#A9B2BD"]
CAT_SHORT = ["Dev./Epi.", "Sig./Trans.", "Horm./Metab.", "Immune", "Cell cycle", "Other"]
OMICS_CMAP = {
    "mRNA": LinearSegmentedColormap.from_list("mRNA", ["#FFFFFF", "#2F6FA3"]),
    "CNV": LinearSegmentedColormap.from_list("CNV", ["#FFFFFF", "#2E8B8B"]),
    "miRNA": LinearSegmentedColormap.from_list("miRNA", ["#FFFFFF", "#B87A1E"]),
}
JSD_CMAP = "magma"


def fill_grid(values_by_pos, size):
    grid = np.zeros((size, size), dtype=float)
    for pos, val in values_by_pos.items():
        grid[pos] = val
    return grid


def load_example(omics, size, sample_idx=0):
    final_df = pd.read_csv(PAM_DIR / f"{omics}_PAM50_final.tsv", sep="\t")
    order = pd.read_csv(IMG_PAM / omics / "order.tsv", sep="\t")
    sample = final_df.iloc[sample_idx, 1:].to_numpy(dtype=float)
    positions = spiral_order(size)
    pos_gene = {pos: gene for pos, gene in zip(positions[: len(order)], order["feature"].tolist())}
    gene_jsd = order.set_index("feature")["jsd"].to_dict()
    expr = {pos: float(sample[final_df.columns.get_loc(pos_gene[pos]) - 1]) for pos in pos_gene}
    jsd = {pos: float(gene_jsd[pos_gene[pos]]) for pos in pos_gene}
    return fill_grid(expr, size), fill_grid(jsd, size)


def category_grid_for(omics):
    if omics == "mRNA":
        return np.load(DATA / "images" / "examples" / "mRNA_category_grid.npy")
    return np.load(IMG_PAM / omics / "category_grid.npy")


def add_coordinate_axes(ax, size):
    if size == 8:
        ticks = [0, 4, 8]
    elif size == 20:
        ticks = [0, 5, 10, 15, 20]
    else:
        ticks = [0, 5, 10, 15, 20, 25]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis="both", labelsize=6.5, length=2.5, width=0.8, colors="#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def set_title(ax, label, text):
    ax.set_title(f"{label}  {text}", loc="left", fontsize=9, fontweight="bold", color="#1F4D78")


def frame_only(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("#7F7F7F")


def main():
    mRNA_expr, mRNA_jsd = load_example("mRNA", 20)
    CNV_expr, _ = load_example("CNV", 8)
    miRNA_expr, _ = load_example("miRNA", 25)
    mRNA_cat = category_grid_for("mRNA")

    fig = plt.figure(figsize=(17.0, 3.9), dpi=300)
    gs = gridspec.GridSpec(
        1,
        5,
        left=0.03,
        right=0.93,
        top=0.86,
        bottom=0.10,
        wspace=0.16,
        width_ratios=[1.0, 1.0, 1.0, 1.06, 1.06],
    )
    axes = np.empty((1, 5), dtype=object)
    for c in range(5):
        axes[0, c] = fig.add_subplot(gs[0, c])

    # Grayscale expression maps for the three omics.
    axes[0, 0].imshow(mRNA_expr, cmap=OMICS_CMAP["mRNA"], vmin=mRNA_expr.min(), vmax=mRNA_expr.max(), aspect="equal", extent=(0, 20, 20, 0))
    set_title(axes[0, 0], "B1", "Expression\n(mRNA, 20×20)")
    frame_only(axes[0, 0])

    axes[0, 1].imshow(CNV_expr, cmap=OMICS_CMAP["CNV"], vmin=CNV_expr.min(), vmax=CNV_expr.max(), aspect="equal", extent=(0, 8, 8, 0))
    set_title(axes[0, 1], "B2", "Expression\n(CNV, 8×8)")
    frame_only(axes[0, 1])

    axes[0, 2].imshow(miRNA_expr, cmap=OMICS_CMAP["miRNA"], vmin=miRNA_expr.min(), vmax=miRNA_expr.max(), aspect="equal", extent=(0, 25, 25, 0))
    set_title(axes[0, 2], "B3", "Expression\n(miRNA, 25×25)")
    frame_only(axes[0, 2])

    # JSD importance map for mRNA.
    im = axes[0, 3].imshow(mRNA_jsd, cmap=JSD_CMAP, vmin=0, vmax=float(mRNA_jsd.max()), aspect="equal", extent=(0, 20, 20, 0))
    set_title(axes[0, 3], "B4", "JSD importance\n(mRNA)")
    add_coordinate_axes(axes[0, 3], 20)
    divider = make_axes_locatable(axes[0, 3])
    cax = divider.append_axes("right", size="6%", pad=0.06)
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=5.5)

    # Functional category map for mRNA with an integrated colorbar/legend.
    cmap = ListedColormap(CAT_COLORS_MID)
    norm = BoundaryNorm(np.arange(-0.5, 6, 1), len(CAT_COLORS))
    im = axes[0, 4].imshow(mRNA_cat, cmap=cmap, norm=norm, aspect="equal", extent=(0, 20, 20, 0))
    set_title(axes[0, 4], "B5", "Gene functional\ncategory (mRNA)")
    add_coordinate_axes(axes[0, 4], 20)
    divider = make_axes_locatable(axes[0, 4])
    cax = divider.append_axes("right", size="9%", pad=0.06)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([i + 0.5 for i in range(6)])
    cbar.set_ticklabels(CAT_SHORT)
    cbar.ax.tick_params(labelsize=5.5, rotation=0)
    cbar.set_label("Functional category", fontsize=7)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
