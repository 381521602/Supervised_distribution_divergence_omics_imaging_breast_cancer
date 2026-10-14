# -*- coding: utf-8 -*-
"""Finalize the rebuilt CNV matrix: key by case_id, deduplicate genes, save."""
import sys
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent.parent / "data"
RAW = D / "cnv_gene_matrix.tsv"
LABELS = D / "brca_labels_modeling_ready.tsv"
OUT = D / "cnv_gene_case_matrix.tsv"


def main():
    mat = pd.read_csv(RAW, sep="\t", index_col=0)
    labels = pd.read_csv(LABELS, sep="\t")
    samp2case = labels.set_index(labels["cnv_gene_sample_id"].str[:15])["case_id"].to_dict()
    mat.index = [samp2case.get(s, s[:12]) for s in mat.index]
    # deduplicate case_id (first occurrence)
    mat = mat[~mat.index.duplicated(keep="first")]
    # deduplicate gene columns (first occurrence)
    mat = mat.loc[:, ~mat.columns.duplicated(keep="first")]
    mat.index.name = "case_id"
    mat.to_csv(OUT, sep="\t")
    print("saved:", OUT, mat.shape)
    print("PAM50 cases present:", mat.index.isin(
        labels[labels["pam50_4class"].isin(["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"])]["case_id"]).sum())
    return 0


if __name__ == "__main__":
    sys.exit(main())
