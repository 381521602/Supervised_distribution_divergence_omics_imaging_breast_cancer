#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate JSD-based feature-mining visualization."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import spiral_order  # noqa: E402


DATA = ROOT / "data"
IMG = DATA / "images/PAM50/mRNA"
INTERP = DATA / "interpretability"
OUT = INTERP / "figures/jsd_feature_mining_visualization.png"


def fill_grid(values_by_pos, size=20):
    grid = np.zeros((size, size), dtype=float)
    for pos, val in values_by_pos.items():
        grid[pos] = val
    return grid


def main():
    df = pd.read_csv(ROOT / "data/final_datasets/PAM50/mRNA_PAM50_final.tsv", sep="\t")
    sample = df.iloc[0, 1:].to_numpy(dtype=float)
    order = pd.read_csv(IMG / "order.tsv", sep="\t")
    positions = spiral_order(20)
    pos_gene = {pos: gene for pos, gene in zip(positions[: len(order)], order["feature"].tolist())}
    gene_jsd = order.set_index("feature")["jsd"].to_dict()

    expr = {pos: float(sample[df.columns.get_loc(pos_gene[pos]) - 1]) for pos in pos_gene}
    jsd = {pos: float(gene_jsd[pos_gene[pos]]) for pos in pos_gene}

    shap = pd.read_csv(INTERP / "mrna_pam50_shap_importance.tsv", sep="\t").set_index("gene")["mean_importance"].to_dict()
    cnn = pd.read_csv(INTERP / "mrna_pam50_cnn_saliency_importance.tsv", sep="\t").set_index("gene")["mean_importance"].to_dict()
    shap_map = {pos: float(shap.get(pos_gene[pos], 0.0)) for pos in pos_gene}
    cnn_map = {pos: float(cnn.get(pos_gene[pos], 0.0)) for pos in pos_gene}

    cat_grid = np.load(DATA / "images/examples/mRNA_category_grid.npy")
    gray = fill_grid(expr)
    jsd_grid = fill_grid(jsd)
    shap_grid = fill_grid(shap_map)
    cnn_grid = fill_grid(cnn_map)

    vmin, vmax = gray.min(), gray.max()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    im0 = axes[0, 0].imshow(gray, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    axes[0, 0].set_title("JSD grayscale expression")
    im1 = axes[0, 1].imshow(jsd_grid, cmap="magma", aspect="equal")
    axes[0, 1].set_title("JSD importance map")
    cat_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]
    cat_names = {
        0: "Development/Epithelium",
        1: "Signaling/Transport",
        2: "Hormone/Metabolism",
        3: "Immune/Inflammation",
        4: "Cell cycle/Proliferation",
        5: "Other",
    }
    cat_norm = BoundaryNorm(np.arange(-0.5, 6, 1), len(cat_colors))
    im_cat = axes[0, 2].imshow(cat_grid, cmap=ListedColormap(cat_colors), norm=cat_norm, aspect="equal")
    axes[0, 2].set_title("Gene functional category map")
    im3 = axes[1, 0].imshow(shap_grid, cmap="viridis", aspect="equal"); axes[1, 0].set_title("SHAP feature importance map")
    im4 = axes[1, 1].imshow(cnn_grid, cmap="cividis", aspect="equal"); axes[1, 1].set_title("CNN saliency feature map")
    ax = axes[1, 2]
    ax.imshow(gray, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    ax.imshow(cnn_grid, cmap="hot", alpha=0.65, aspect="equal")
    ax.set_title("Overlay: expression + CNN saliency")
    for axx in axes.ravel():
        axx.axis("off")
    for im, axx in [(im0, axes[0, 0]), (im1, axes[0, 1]), (im_cat, axes[0, 2]), (im3, axes[1, 0]), (im4, axes[1, 1])]:
        divider = make_axes_locatable(axx)
        cax = divider.append_axes("right", size="4%", pad=0.12)
        cbar = fig.colorbar(im, cax=cax)
        if im is im_cat:
            cbar.set_ticks([i + 0.5 for i in range(6)])
            cbar.set_ticklabels([cat_names[i].split("/")[0] for i in range(6)])
            cbar.ax.tick_params(labelsize=10)
    fig.suptitle("JSD image ordering for interpretable feature mining", fontsize=14)
    fig.subplots_adjust(left=0.05, right=0.90, top=0.90, bottom=0.12, wspace=0.28, hspace=0.28)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved ->", OUT)


if __name__ == "__main__":
    main()
