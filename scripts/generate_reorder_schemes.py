#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按三种方案重新规划 PAM50 mRNA 图像基因顺序，并生成单样本三联图。

方案1：功能大类分块 + 块内 NSRE 降序，行优先填充
方案2：功能大类分块 + 中心高重要性（块均 NSRE 排序，中心向外螺旋填充）
方案3：样本表达相关性层次聚类排序，行优先填充
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import spiral_order  # noqa: E402


DATA = ROOT / "data"
PAM_DIR = DATA / "final_datasets/PAM50"
ORIG_IMG_DIR = DATA / "images/PAM50/mRNA"
OUT_BASE = DATA / "images/PAM50/mRNA_reorder"
EXAMPLE_DIR = DATA / "images/examples"

CATEGORIES = {
    0: "Development/Epithelium",
    1: "Signaling/Transport",
    2: "Hormone/Metabolism",
    3: "Immune/Inflammation",
    4: "Cell cycle/Proliferation",
    5: "Other",
}
CAT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]


def row_major_positions(size: int):
    return [(r, c) for r in range(size) for c in range(size)]


def recover_gene_categories(order_idx: np.ndarray) -> np.ndarray:
    category_grid = np.load(DATA / "images/examples/mRNA_category_grid.npy")
    positions = spiral_order(20)
    ordered_cat = np.array([category_grid[pos] for pos in positions[:400]], dtype=int)
    cat = np.zeros(400, dtype=int)
    cat[order_idx] = ordered_cat
    return cat


def load_gene_metadata(df: pd.DataFrame):
    feature_names = df.columns[1:].tolist()
    order_tsv = pd.read_csv(ORIG_IMG_DIR / "order.tsv", sep="\t")
    nsre = order_tsv.set_index("feature")["nsre"].to_dict()
    return feature_names, nsre


def make_images(X: np.ndarray, size: int, order: np.ndarray, positions) -> np.ndarray:
    n = X.shape[0]
    images = np.zeros((n, 1, size, size), dtype=np.float32)
    for i in range(n):
        img = np.zeros((size, size), dtype=np.float32)
        for (r, c), j in zip(positions, order):
            img[r, c] = X[i, j]
        images[i, 0] = img
    return images


def make_grids(size, order, positions, cat, nsre_by_idx):
    cat_grid = np.zeros((size, size), dtype=int)
    nsre_grid = np.zeros((size, size), dtype=float)
    for (r, c), j in zip(positions, order):
        cat_grid[r, c] = cat[j]
        nsre_grid[r, c] = nsre_by_idx[j]
    return cat_grid, nsre_grid


def save_scheme(name, order, positions, X, cat, nsre_by_idx, feature_names, nsre_map):
    out = OUT_BASE / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "order.npy", order)
    images = make_images(X, 20, order, positions)
    np.save(out / "images.npy", images)
    cat_grid, nsre_grid = make_grids(20, order, positions, cat, nsre_by_idx)
    np.save(out / "category_grid.npy", cat_grid)
    np.save(out / "nsre_grid.npy", nsre_grid)
    pd.DataFrame(
        {
            "feature": [feature_names[j] for j in order],
            "nsre": [nsre_map[feature_names[j]] for j in order],
        }
    ).to_csv(out / "order.tsv", sep="\t", index=False)
    print(f"Saved scheme data -> {out}")


def draw_triple(name, order, positions, X, cat, nsre_by_idx, sample_idx=0):
    size = 20
    sample = X[sample_idx]
    vmin = float(np.min(X))
    vmax = float(np.max(X))
    gray = np.zeros((size, size), dtype=float)
    cat_grid = np.zeros((size, size), dtype=int)
    nsre_grid = np.zeros((size, size), dtype=float)
    for (r, c), j in zip(positions, order):
        gray[r, c] = sample[j]
        cat_grid[r, c] = cat[j]
        nsre_grid[r, c] = nsre_by_idx[j]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    im0 = axes[0].imshow(gray, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    axes[0].set_title("Grayscale expression", fontsize=10)
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    cmap = matplotlib.colors.ListedColormap(CAT_COLORS)
    im1 = axes[1].imshow(cat_grid, cmap=cmap, vmin=0, vmax=5, aspect="equal")
    axes[1].set_title("Gene functional category", fontsize=10)
    axes[1].axis("off")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, ticks=range(6))
    cbar1.set_ticklabels([CATEGORIES[i].split("/")[0] for i in range(6)])
    cbar1.ax.tick_params(labelsize=7)

    im2 = axes[2].imshow(nsre_grid, cmap="magma", vmin=0, vmax=float(np.nanmax(nsre_grid)), aspect="equal")
    axes[2].set_title("NSRE score", fontsize=10)
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"PAM50 mRNA reordering: {name} (sample {sample_idx})", fontsize=12)
    out = EXAMPLE_DIR / f"PAM50_mRNA_{name}_triple.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved triple figure -> {out}")


def main():
    df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    feature_names, nsre_map = load_gene_metadata(df)
    nsre_by_idx = np.array([nsre_map[g] for g in feature_names], dtype=float)
    orig_order = np.load(ORIG_IMG_DIR / "order.npy")
    cat = recover_gene_categories(orig_order)

    # 方案1：功能类别顺序，块内 NSRE 降序，行优先
    block_order1 = []
    for c in sorted(set(cat.tolist())):
        idx = np.where(cat == c)[0]
        idx = idx[np.argsort(nsre_by_idx[idx])[::-1]]
        block_order1.extend(idx.tolist())
    order1 = np.asarray(block_order1, dtype=int)
    positions1 = row_major_positions(20)
    save_scheme("scheme1_function_block", order1, positions1, X, cat, nsre_by_idx, feature_names, nsre_map)
    draw_triple("scheme1_function_block", order1, positions1, X, cat, nsre_by_idx)

    # 方案2：功能块按平均 NSRE 排序，中心向外螺旋
    blocks = []
    for c in set(cat.tolist()):
        idx = np.where(cat == c)[0]
        idx = idx[np.argsort(nsre_by_idx[idx])[::-1]]
        blocks.append((float(np.mean(nsre_by_idx[idx])), c, idx.tolist()))
    blocks.sort(key=lambda t: t[0], reverse=True)
    order2 = np.asarray([j for _, _, idx in blocks for j in idx], dtype=int)
    positions2 = spiral_order(20)
    save_scheme("scheme2_function_center", order2, positions2, X, cat, nsre_by_idx, feature_names, nsre_map)
    draw_triple("scheme2_function_center", order2, positions2, X, cat, nsre_by_idx)

    # 方案3：表达相关性层次聚类
    corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order3 = np.asarray(leaves_list(Z), dtype=int)
    positions3 = row_major_positions(20)
    save_scheme("scheme3_expression_cluster", order3, positions3, X, cat, nsre_by_idx, feature_names, nsre_map)
    draw_triple("scheme3_expression_cluster", order3, positions3, X, cat, nsre_by_idx)

    print("All reorder schemes completed.")


if __name__ == "__main__":
    main()
