# -*- coding: utf-8 -*-
"""CoxNet and Random Survival Forest survival comparison + time-dependent AUC."""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
XENA = D / "external" / "xena"
GH = ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_jsd import build_adaptive_pipeline

LABELS = GH / "data" / "brca_labels_modeling_ready.tsv"
OUT = ROOT / "data" / "leakfree_survival_methods.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
OMICS = ["mRNA", "CNV", "miRNA"]
SURV_K = {"mRNA": 200, "CNV": 150, "miRNA": 600}


def load_surv():
    labels = pd.read_csv(LABELS, sep="\t")
    labels = labels[labels["os_event"].isin([0, 1]) & labels["os_time_days"].notna()]
    data = {}
    for o, (fname, col) in {
        "mRNA": (XENA / "HiSeqV2", "mrna_sample_id"),
        "miRNA": (XENA / "miRNA_HiSeq_gene", "mirna_sample_id"),
    }.items():
        raw = pd.read_csv(fname, sep="\t", index_col=0).T
        lab = labels.dropna(subset=[col]).drop_duplicates(subset=["case_id"]).copy()
        lab["prefix"] = lab[col].str[:15]
        idx = [s for s in lab["prefix"] if s in raw.index]
        lab2 = lab.set_index("prefix").loc[idx]
        X = np.nan_to_num(raw.loc[idx].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        data[o] = (X, lab2["case_id"].tolist())
    cnv = pd.read_csv(D / "cnv_gene_case_matrix.tsv", sep="\t", index_col=0)
    idx = [c for c in labels["case_id"] if c in cnv.index]
    X = np.nan_to_num(cnv.loc[idx].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    data["CNV"] = (X, idx)
    return data, labels.set_index("case_id")


def main():
    data, labels = load_surv()
    y = labels["os_event"].astype(bool).values
    t = labels["os_time_days"].astype(float).values
    surv = Surv.from_arrays(event=y, time=t)
    rows = []
    tdauc_rows = []
    for o in OMICS:
        X, cases = data[o]
        lab = labels.loc[cases]
        ye = lab["os_event"].astype(bool).values
        yt = lab["os_time_days"].astype(float).values
        cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        ci_coxnet, ci_rsf = [], []
        preds = np.zeros(len(ye))
        for tr, te in cv.split(np.zeros(len(ye)), ye.astype(int)):
            pipe = build_adaptive_pipeline(o, "OS", k=SURV_K[o]).fit(X[tr], ye[tr].astype(int))
            Xtr = pipe.transform(X[tr]); Xte = pipe.transform(X[te])
            ytr = Surv.from_arrays(event=ye[tr], time=yt[tr])
            cn = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=10000).fit(Xtr, ytr)
            r = cn.predict(Xte)
            ci_coxnet.append(concordance_index_censored(ye[te], yt[te], r)[0])
            rsf = RandomSurvivalForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr, ytr)
            r2 = rsf.predict(Xte)
            ci_rsf.append(concordance_index_censored(ye[te], yt[te], r2)[0])
            preds[te] = r  # CoxNet risk (higher = higher risk)
            print(f"[{o}] fold done", flush=True)
        rows.append({"omics": o, "model": "CoxNet", "metric": "c_index",
                     "mean": round(float(np.mean(ci_coxnet)), 4), "std": round(float(np.std(ci_coxnet)), 4)})
        rows.append({"omics": o, "model": "RSF", "metric": "c_index",
                     "mean": round(float(np.mean(ci_rsf)), 4), "std": round(float(np.std(ci_rsf)), 4)})
        # time-dependent AUC (CoxNet out-of-fold risk)
        try:
            surv_test = Surv.from_arrays(event=ye, time=yt)
            times = np.quantile(yt[ye], [0.5, 0.75, 1.0])
            auc, _ = cumulative_dynamic_auc(surv_test, surv_test, preds, times)
            tdauc_rows.append({"omics": o, "metric": "time_dependent_auc",
                               "mean": round(float(np.nanmean(auc)), 4),
                               "times": ",".join(str(round(float(x), 0)) for x in times)})
        except Exception as e:
            tdauc_rows.append({"omics": o, "metric": "time_dependent_auc", "mean": None, "times": str(e)[:50]})
    all_rows = rows + tdauc_rows
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["omics", "model", "metric", "mean", "std", "times"], delimiter="\t")
        w.writeheader()
        for r in all_rows:
            r.setdefault("std", "")
            r.setdefault("times", "")
            w.writerow(r)
    print("done ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
