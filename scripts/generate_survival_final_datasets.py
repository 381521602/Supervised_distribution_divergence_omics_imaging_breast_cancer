#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate final survival feature matrices using optimal selectors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_batch1_pam50_grid import make_selector, load_omics  # noqa: E402


OUT_DIR = ROOT / "data" / "final_datasets" / "Survival"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "mRNA": ("V_NSRE_F", 200),
    "CNV": ("V_L1_NSRE", 150),
    "miRNA": ("V_L1_NSRE", 600),
}


def align_os(omics):
    matrix = load_omics(omics)
    labels = pd.read_csv(ROOT/"data/brca_labels_modeling_ready.tsv", sep="\t", dtype={"case_id": str})
    cases = labels.loc[labels["os_event"].isin([0,1]) & labels["os_time_days"].notna(), "case_id"].tolist()
    case_cols = {}
    for col in matrix.columns:
        case_cols.setdefault(col[:12], []).append(col)
    best_cols = {c: ([x for x in cols if x.endswith("-01")] or cols)[0] for c, cols in case_cols.items()}
    aligned = [c for c in cases if c in best_cols]
    X = matrix[[best_cols[c] for c in aligned]].T.to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    lab = labels.set_index("case_id").loc[aligned]
    return matrix, X, aligned, lab


def main():
    for omics, (fs, k) in CONFIG.items():
        matrix, X, cases, lab = align_os(omics)
        sel = make_selector(fs, k)
        sel.fit(X, lab["os_event"].astype(int).values)
        X_sel = sel.transform(X)
        indices = np.arange(X.shape[1])
        for step_name, step in sel.steps:
            if hasattr(step, "selected_features_"):
                indices = indices[step.selected_features_]
            elif hasattr(step, "get_support"):
                indices = indices[step.get_support(indices=True)]
        feature_names = [str(matrix.index[i]) for i in indices]
        out = pd.DataFrame(X_sel, columns=feature_names)
        out.insert(0, "case_id", cases)
        out.insert(1, "os_event", lab["os_event"].astype(int).values)
        out.insert(2, "os_time_days", lab["os_time_days"].astype(float).values)
        out.to_csv(OUT_DIR / f"{omics}_Survival_final.tsv", sep="\t", index=False)
        pd.DataFrame({"feature": feature_names}).to_csv(OUT_DIR / f"{omics}_Survival_features.tsv", sep="\t", index=False)
        print(f"{omics}: {X_sel.shape[0]} x {X_sel.shape[1]}")
    print("Done")


if __name__ == "__main__":
    main()
