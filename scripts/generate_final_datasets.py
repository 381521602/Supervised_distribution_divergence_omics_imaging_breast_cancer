#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate final PAM50 feature matrices using optimal selectors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_batch1_pam50_grid import align, make_selector, load_omics  # noqa: E402


OUT_DIR = ROOT / "data" / "final_datasets" / "PAM50"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "mRNA": ("V_L1", 400),
    "CNV": ("V_L1_F", 50),
    "miRNA": ("V_F_JSD", 600),
}


def main() -> None:
    for omics, (fs, k) in CONFIG.items():
        X, y = align(omics)
        sel = make_selector(fs, k)
        sel.fit(X, y)
        X_sel = sel.transform(X)
        support = None
        # recover selected feature names via final step support where possible
        indices = np.arange(X.shape[1])
        for step_name, step in sel.steps:
            if hasattr(step, "selected_features_"):
                indices = indices[step.selected_features_]
            elif hasattr(step, "get_support"):
                indices = indices[step.get_support(indices=True)]
        feature_names = [str(load_omics(omics).index[i]) for i in indices]
        reduced = pd.DataFrame(X_sel, columns=feature_names)
        reduced.insert(0, "pam50", y)
        reduced.to_csv(OUT_DIR / f"{omics}_PAM50_final.tsv", sep="\t", index=False)
        pd.DataFrame({"feature": feature_names}).to_csv(OUT_DIR / f"{omics}_PAM50_features.tsv", sep="\t", index=False)
        print(f"{omics}: {X_sel.shape[0]} samples x {X_sel.shape[1]} features -> {OUT_DIR}")


if __name__ == "__main__":
    main()
