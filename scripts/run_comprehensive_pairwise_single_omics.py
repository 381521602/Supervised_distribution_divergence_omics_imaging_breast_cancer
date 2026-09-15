#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-omics baselines on each pairwise-omics intersection."""

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
OUT = DATA / "comprehensive_pairwise_single_omics_results.tsv"
PAIRS = [("mRNA", "CNV"), ("mRNA", "miRNA"), ("CNV", "miRNA")]
SIZE_PAM = {"mRNA": 20, "CNV": 8, "miRNA": 25}
SIZE_SUR = {"mRNA": 15, "CNV": 13, "miRNA": 25}


def common_cases(task, pair):
    suffix = "PAM50_4class_matrix.tsv" if task == "PAM50" else "OS_matrix.tsv"
    lists = [pd.read_csv(SEL / f"{o}_{suffix}", sep="\t", usecols=[0], index_col=0).index.tolist() for o in pair]
    return [c for c in lists[0] if c in set(lists[1])], lists


def main():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    rows = []

    for pair in PAIRS:
        common, lists = common_cases("PAM50", pair)
        for o, cases in zip(pair, lists):
            df = pd.read_csv(PAM_DIR / f"{o}_PAM50_final.tsv", sep="\t")
            Xmat = df.drop(columns=["pam50"]).to_numpy(dtype=float)
            pos = {c: i for i, c in enumerate(cases)}
            idx = np.array([pos[c] for c in common], dtype=int)
            X = Xmat[idx]
            imgs = np.load(IMG / "PAM50" / o / "images.npy")[idx]
            y_raw = labels.set_index("case_id").loc[common, "pam50_4class"].values
            enc = LabelEncoder()
            y = enc.fit_transform(y_raw)
            for r in eval_pam50(o, X, y, imgs, SIZE_PAM[o]):
                r["pair"] = "+".join(pair)
                rows.append(r)
        print(f"done PAM50 {pair}", flush=True)

    for pair in PAIRS:
        common, lists = common_cases("Survival", pair)
        for o, cases in zip(pair, lists):
            df = pd.read_csv(SUR_DIR / f"{o}_Survival_final.tsv", sep="\t")
            Xmat = df.drop(columns=["case_id", "os_event", "os_time_days"]).to_numpy(dtype=float)
            pos = {c: i for i, c in enumerate(cases)}
            idx = np.array([pos[c] for c in common], dtype=int)
            X = Xmat[idx]
            imgs = np.load(IMG / "Survival" / o / "images.npy")[idx]
            lab = labels.set_index("case_id").loc[common]
            y_event = lab["os_event"].astype(int).values
            y_time = lab["os_time_days"].astype(float).values
            for r in eval_survival(o, X, y_event, y_time, imgs, SIZE_SUR[o]):
                r["pair"] = "+".join(pair)
                rows.append(r)
        print(f"done Survival {pair}", flush=True)

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "omics", "task", "model", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
