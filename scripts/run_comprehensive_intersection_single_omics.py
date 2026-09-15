#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-omics baselines on the three-omics intersection."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_comprehensive_batch1_single_omics import eval_pam50, eval_survival  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
SEL = DATA / "selected_features"
PAM_DIR = DATA / "final_datasets/PAM50"
SUR_DIR = DATA / "final_datasets/Survival"
IMG = DATA / "images"
OUT = DATA / "comprehensive_intersection_single_omics_results.tsv"
OMICS = ["mRNA", "CNV", "miRNA"]
SIZE_PAM = {"mRNA": 20, "CNV": 8, "miRNA": 25}
SIZE_SUR = {"mRNA": 15, "CNV": 13, "miRNA": 25}


def pam_common():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    case_lists = [pd.read_csv(SEL / f"{o}_PAM50_4class_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in OMICS]
    common = [c for c in case_lists[0] if c in set(case_lists[1]) and c in set(case_lists[2])]
    return labels, case_lists, common


def sur_common():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    case_lists = [pd.read_csv(SEL / f"{o}_OS_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in OMICS]
    common = [c for c in case_lists[0] if c in set(case_lists[1]) and c in set(case_lists[2])]
    return labels, case_lists, common


def main():
    rows = []
    labels, case_lists, common = pam_common()
    for o, cases in zip(OMICS, case_lists):
        df = pd.read_csv(PAM_DIR / f"{o}_PAM50_final.tsv", sep="\t")
        Xmat = df.drop(columns=["pam50"]).to_numpy(dtype=float)
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X = Xmat[idx]
        imgs = np.load(IMG / "PAM50" / o / "images.npy")[idx]
        y_raw = labels.set_index("case_id").loc[common, "pam50_4class"].values
        enc = LabelEncoder()
        y = enc.fit_transform(y_raw)
        rows += eval_pam50(o, X, y, imgs, SIZE_PAM[o])
        print(f"done PAM50 {o}", flush=True)

    labels, case_lists, common = sur_common()
    for o, cases in zip(OMICS, case_lists):
        df = pd.read_csv(SUR_DIR / f"{o}_Survival_final.tsv", sep="\t")
        Xmat = df.drop(columns=["case_id", "os_event", "os_time_days"]).to_numpy(dtype=float)
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X = Xmat[idx]
        imgs = np.load(IMG / "Survival" / o / "images.npy")[idx]
        lab = labels.set_index("case_id").loc[common]
        y_event = lab["os_event"].astype(int).values
        y_time = lab["os_time_days"].astype(float).values
        rows += eval_survival(o, X, y_event, y_time, imgs, SIZE_SUR[o])
        print(f"done Survival {o}", flush=True)

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["omics", "task", "model", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
