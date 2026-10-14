# -*- coding: utf-8 -*-
"""Regenerate the single-sample mRNA functional-category maps with the
corrected category assignment (in JSD order)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"E:\组学数据图像化")
GH = ROOT / "github_release"
D = GH / "data"
sys.path.insert(0, str(ROOT / "scripts"))
from fix_mrna_categories import assign, CATEGORY_NAMES  # noqa: E402
sys.path.insert(0, str(GH / "scripts"))
from run_multistream_cnn import spiral_order  # noqa: E402

CAT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]


def main():
    for task, size, out, order_path in [
        ("PAM50", 20, D / "images/examples/mRNA_gene_category_map.png", D / "images/PAM50/mRNA/order.tsv"),
        ("Survival", 15, D / "images/examples/Survival_mRNA_gene_category_map.png", D / "images/Survival/mRNA/order.tsv"),
    ]:
        order_tsv = pd.read_csv(order_path, sep="\t")
        feats = order_tsv["feature"].astype(str).tolist()
        cat = np.array([assign(f) for f in feats], dtype=int)
        grid = np.full((size, size), 5, dtype=int)
        positions = spiral_order(size)
        for (r, c), j in zip(positions, range(len(feats))):
            grid[r, c] = cat[j]
        cmap = matplotlib.colors.ListedColormap(CAT_COLORS)
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=5, aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=CAT_COLORS[i], label=CATEGORY_NAMES[i]) for i in range(6)]
        ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("saved:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
