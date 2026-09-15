#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export adaptive NSRE-selected features per omics for downstream fusion."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from adaptive_nsre import select_features  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
OUT_DIR = DATA / "selected_features"
MATRICES = {
    "mRNA": DATA / "external" / "xena" / "HiSeqV2",
    "CNV": DATA / "external" / "xena" / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
    "miRNA": DATA / "external" / "xena" / "miRNA_HiSeq_gene",
}
TASKS = ["PAM50_4class", "OS"]
FEATURE_COUNTS = {"mRNA": 200, "CNV": 50, "miRNA": 200}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    summary: list[dict] = []

    for omics, path in MATRICES.items():
        if not path.exists():
            print(f"Skip {omics}: missing {path}")
            continue
        matrix = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
        matrix = matrix.loc[:, ~matrix.columns.duplicated()]

        for task in TASKS:
            k = FEATURE_COUNTS[omics]
            feature_names, reduced = select_features(matrix, labels, omics, task, k=k)
            prefix = f"{omics}_{task}"
            feature_path = OUT_DIR / f"{prefix}_features.tsv"
            matrix_path = OUT_DIR / f"{prefix}_matrix.tsv"

            pd.DataFrame({"feature": feature_names}).to_csv(
                feature_path, sep="\t", index=False
            )
            reduced.to_csv(matrix_path, sep="\t")

            summary.append(
                {
                    "omics": omics,
                    "task": task,
                    "k_features": k,
                    "n_features": len(feature_names),
                    "n_samples": reduced.shape[0],
                    "features_file": feature_path.name,
                    "matrix_file": matrix_path.name,
                }
            )
            print(f"{prefix}: {reduced.shape[0]} samples x {len(feature_names)} features")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "selected_features_summary.tsv", sep="\t", index=False)
    print(f"Summary -> {OUT_DIR / 'selected_features_summary.tsv'}")


if __name__ == "__main__":
    main()
