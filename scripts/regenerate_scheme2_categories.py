# -*- coding: utf-8 -*-
"""Regenerate the scheme-2 mRNA category_grid.npy with the corrected category
assignment (same gene order, updated category labels)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\组学数据图像化")
GH = ROOT / "github_release"
D = GH / "data"
sys.path.insert(0, str(ROOT / "scripts"))
from fix_mrna_categories import assign, CATEGORY_NAMES  # noqa: E402


def clean(f):
    return f.split("|")[0].strip().upper()


def main():
    for task in ["PAM50", "Survival"]:
        base = D / "images" / task / "mRNA_reorder" / "scheme2_function_center"
        order_tsv = pd.read_csv(base / "order.tsv", sep="\t")
        feats = order_tsv["feature"].astype(str).tolist()
        size = 20 if task == "PAM50" else 15
        cat = np.array([assign(f) for f in feats], dtype=int)
        # build category_grid by reading the existing positions
        positions = []
        # reuse spiral order (import) to place genes
        sys.path.insert(0, str(GH / "scripts"))
        from run_multistream_cnn import spiral_order
        positions = spiral_order(size)
        grid = np.zeros((size, size), dtype=int)
        for (r, c), j in zip(positions, range(len(feats))):
            grid[r, c] = cat[j]
        np.save(base / "category_grid.npy", grid)
        print(f"{task}: category_grid regenerated, counts={dict(pd.Series(cat).map(CATEGORY_NAMES).value_counts())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
