#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按三种方案重新规划 Survival mRNA 图像基因顺序。

方案1：功能大类分块 + 块内 JSD 降序，行优先填充
方案2：功能大类分块 + 中心高重要性，中心向外螺旋填充
方案3：样本表达相关性层次聚类排序，行优先填充
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import spiral_order  # noqa: E402


DATA = ROOT / "data"
SUR_DIR = DATA / "final_datasets/Survival"
ORIG_IMG_DIR = DATA / "images/Survival/mRNA"
OUT_BASE = DATA / "images/Survival/mRNA_reorder"
SIZE = 15


def row_major_positions(size: int):
    return [(r, c) for r in range(size) for c in range(size)]


def recover_gene_categories(order_idx: np.ndarray) -> np.ndarray:
    category_grid = np.load(DATA / "images/examples/Survival_mRNA_category_grid.npy")
    positions = spiral_order(SIZE)
    ordered_cat = np.array([category_grid[pos] for pos in positions[:200]], dtype=int)
    cat = np.zeros(200, dtype=int)
    cat[order_idx] = ordered_cat
    return cat


def make_images(X: np.ndarray, order: np.ndarray, positions) -> np.ndarray:
    n = X.shape[0]
    images = np.zeros((n, 1, SIZE, SIZE), dtype=np.float32)
    for i in range(n):
        img = np.zeros((SIZE, SIZE), dtype=np.float32)
        for (r, c), j in zip(positions, order):
            img[r, c] = X[i, j]
        images[i, 0] = img
    return images


def save_scheme(name, order, positions, X, cat, jsd_by_idx, feature_names, jsd_map):
    out = OUT_BASE / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "order.npy", order)
    np.save(out / "images.npy", make_images(X, order, positions))
    cat_grid = np.zeros((SIZE, SIZE), dtype=int)
    jsd_grid = np.zeros((SIZE, SIZE), dtype=float)
    for (r, c), j in zip(positions, order):
        cat_grid[r, c] = cat[j]
        jsd_grid[r, c] = jsd_by_idx[j]
    np.save(out / "category_grid.npy", cat_grid)
    np.save(out / "jsd_grid.npy", jsd_grid)
    pd.DataFrame(
        {
            "feature": [feature_names[j] for j in order],
            "jsd": [jsd_map[feature_names[j]] for j in order],
        }
    ).to_csv(out / "order.tsv", sep="\t", index=False)
    print(f"Saved survival scheme data -> {out}")


def main():
    df = pd.read_csv(SUR_DIR / "mRNA_Survival_final.tsv", sep="\t")
    X = df.iloc[:, 3:].to_numpy(dtype=np.float32)
    feature_names = df.columns[3:].tolist()
    order_tsv = pd.read_csv(ORIG_IMG_DIR / "order.tsv", sep="\t")
    jsd_map = order_tsv.set_index("feature")["jsd"].to_dict()
    jsd_by_idx = np.array([jsd_map[g] for g in feature_names], dtype=float)
    orig_order = np.load(ORIG_IMG_DIR / "order.npy")
    cat = recover_gene_categories(orig_order)

    # 方案1
    block_order1 = []
    for c in sorted(set(cat.tolist())):
        idx = np.where(cat == c)[0]
        idx = idx[np.argsort(jsd_by_idx[idx])[::-1]]
        block_order1.extend(idx.tolist())
    order1 = np.asarray(block_order1, dtype=int)
    save_scheme("scheme1_function_block", order1, row_major_positions(SIZE), X, cat, jsd_by_idx, feature_names, jsd_map)

    # 方案2
    blocks = []
    for c in set(cat.tolist()):
        idx = np.where(cat == c)[0]
        idx = idx[np.argsort(jsd_by_idx[idx])[::-1]]
        blocks.append((float(np.mean(jsd_by_idx[idx])), c, idx.tolist()))
    blocks.sort(key=lambda t: t[0], reverse=True)
    order2 = np.asarray([j for _, _, idx in blocks for j in idx], dtype=int)
    save_scheme("scheme2_function_center", order2, spiral_order(SIZE), X, cat, jsd_by_idx, feature_names, jsd_map)

    # 方案3
    corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order3 = np.asarray(leaves_list(Z), dtype=int)
    save_scheme("scheme3_expression_cluster", order3, row_major_positions(SIZE), X, cat, jsd_by_idx, feature_names, jsd_map)

    print("All survival reorder schemes completed.")


if __name__ == "__main__":
    main()
