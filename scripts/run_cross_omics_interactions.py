from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/selected_features"
OUT = ROOT / "data/gene_interaction"
OUT.mkdir(parents=True, exist_ok=True)


def read_matrix(path: Path):
    df = pd.read_csv(path, sep="\t")
    df = df.set_index("case_id")
    X = df.to_numpy(dtype=float)
    feats = df.columns.tolist()
    return X, feats


def top_edges(mat, feats1, feats2, kind, n_top=800):
    rows = []
    for i in range(mat.shape[0]):
        vals = mat[i, :]
        if kind == "miRNA_mRNA":
            cand = np.argsort(vals)[: n_top // mat.shape[0] + 1]
            cand = cand[vals[cand] < 0]
        else:
            order = np.argsort(-np.abs(vals))
            cand = order[: n_top // mat.shape[0] + 1]
        for j in cand:
            rows.append({"feature1": feats1[i], "feature2": feats2[j], "rho": round(float(vals[j]), 4)})
    df = pd.DataFrame(rows)
    if kind == "miRNA_mRNA":
        df = df[df["rho"] < 0].sort_values("rho").head(n_top)
    else:
        df["abs_rho"] = df["rho"].abs()
        df = df.sort_values("abs_rho", ascending=False).head(n_top).drop(columns=["abs_rho"])
    return df


def main():
    mrna_df = pd.read_csv(FEAT / "mRNA_PAM50_4class_matrix.tsv", sep="\t").set_index("case_id")
    mirna_df = pd.read_csv(FEAT / "miRNA_PAM50_4class_matrix.tsv", sep="\t").set_index("case_id")
    cnv_df = pd.read_csv(FEAT / "CNV_PAM50_4class_matrix.tsv", sep="\t").set_index("case_id")
    common = mrna_df.index.intersection(mirna_df.index).intersection(cnv_df.index)
    mrna_df = mrna_df.loc[common]
    mirna_df = mirna_df.loc[common]
    cnv_df = cnv_df.loc[common]
    mrna, mrna_feats = mrna_df.to_numpy(dtype=float), mrna_df.columns.tolist()
    mirna, mirna_feats = mirna_df.to_numpy(dtype=float), mirna_df.columns.tolist()
    cnv, cnv_feats = cnv_df.to_numpy(dtype=float), cnv_df.columns.tolist()

    mm = np.zeros((mirna.shape[1], mrna.shape[1]), dtype=float)
    for i in range(mirna.shape[1]):
        for j in range(mrna.shape[1]):
            rho, _ = spearmanr(mirna[:, i], mrna[:, j])
            mm[i, j] = rho if np.isfinite(rho) else 0.0
    mm_df = top_edges(mm, mirna_feats, mrna_feats, "miRNA_mRNA")
    mm_df.to_csv(OUT / "miRNA_mRNA_negative_correlations.tsv", sep="\t", index=False)

    cm = np.zeros((cnv.shape[1], mrna.shape[1]), dtype=float)
    for i in range(cnv.shape[1]):
        for j in range(mrna.shape[1]):
            rho, _ = spearmanr(cnv[:, i], mrna[:, j])
            cm[i, j] = rho if np.isfinite(rho) else 0.0
    cm_df = top_edges(cm, cnv_feats, mrna_feats, "CNV_mRNA")
    cm_df.to_csv(OUT / "cnv_mRNA_correlations.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    top_mirna = sorted(set(mm_df["feature1"]))[:20]
    top_mrna = sorted(set(mm_df["feature2"]))[:30]
    mi_idx = [mirna_feats.index(g) for g in top_mirna]
    mr_idx = [mrna_feats.index(g) for g in top_mrna]
    ax = axes[0]
    im = ax.imshow(mm[np.ix_(mi_idx, mr_idx)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_mrna)), top_mrna, rotation=90, fontsize=5)
    ax.set_yticks(range(len(top_mirna)), top_mirna, fontsize=5)
    ax.set_title("miRNA-mRNA negative correlations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    top_cnv = sorted(set(cm_df["feature1"]))[:20]
    top_cmr = sorted(set(cm_df["feature2"]))[:30]
    ci_idx = [cnv_feats.index(g) for g in top_cnv]
    cr_idx = [mrna_feats.index(g) for g in top_cmr]
    ax = axes[1]
    im = ax.imshow(cm[np.ix_(ci_idx, cr_idx)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_cmr)), top_cmr, rotation=90, fontsize=5)
    ax.set_yticks(range(len(top_cnv)), top_cnv, fontsize=5)
    ax.set_title("CNV-mRNA correlations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cross_omics_interactions.png", dpi=200)
    print("miRNA-mRNA edges", len(mm_df), "CNV-mRNA edges", len(cm_df))
    print("saved to", OUT)


if __name__ == "__main__":
    main()
