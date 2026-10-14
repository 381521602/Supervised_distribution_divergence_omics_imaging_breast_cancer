# -*- coding: utf-8 -*-
"""Rebuild the gene-level CNV matrix from downloaded GDC files and threshold
copy number to GISTIC2-like -2/-1/0/1/2 values."""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(r"E:\组学数据图像化\data")
GDC = D / "gdc_cnv"
OUT = D / "cnv_gene_matrix.tsv"


def threshold(cn):
    if pd.isna(cn):
        return 0
    cn = float(cn)
    if cn <= 0.5:
        return -2
    if cn <= 1.5:
        return -1
    if cn <= 2.5:
        return 0
    if cn <= 3.5:
        return 1
    return 2


def main():
    files = sorted(glob.glob(str(GDC / "*.tsv")))
    print("files:", len(files))
    # gather all gene names from one file (union across files)
    gene_cols = None
    sample_genes = {}
    for f in files:
        sample = os.path.basename(f).split("_")[0]
        df = pd.read_csv(f, sep="\t", usecols=["gene_name", "copy_number"])
        sample_genes[sample] = df
    # union of gene names
    all_genes = set()
    for df in sample_genes.values():
        all_genes.update(df["gene_name"].dropna().unique())
    all_genes = sorted(g for g in all_genes if isinstance(g, str) and g)
    print("union genes:", len(all_genes))
    gidx = {g: i for i, g in enumerate(all_genes)}
    samples = sorted(sample_genes.keys())
    M = np.zeros((len(samples), len(all_genes)), dtype=np.int8)
    for si, s in enumerate(samples):
        df = sample_genes[s]
        for g, cn in zip(df["gene_name"], df["copy_number"]):
            if g in gidx:
                M[si, gidx[g]] = threshold(cn)
    mat = pd.DataFrame(M, index=samples, columns=all_genes)
    mat.index.name = "sample"
    mat.to_csv(OUT, sep="\t")
    print("saved:", OUT, mat.shape)
    # verify against original 50-feature matrix
    orig = pd.read_csv(D / "github_release" / "data" / "selected_features" / "CNV_PAM50_4class_matrix.tsv",
                       sep="\t", index_col=0)
    feats = [c for c in orig.columns if c in mat.columns]
    common = [s for s in orig.index if s in mat.index]
    if feats and common:
        sub = mat.loc[common, feats].reindex(columns=feats)
        osub = orig.loc[common, feats]
        match = float((sub.to_numpy() == osub.to_numpy()).mean())
        print("overlap samples:", len(common), "features:", len(feats))
        print("value match ratio vs original:", round(match, 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
