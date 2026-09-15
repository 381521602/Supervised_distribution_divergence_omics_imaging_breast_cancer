#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 1: PAM50 grid with traditional ML."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from adaptive_nsre import NSRESelector  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
XENA = DATA / "external" / "xena"
OUT = DATA / "batch1_pam50_grid_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
OMICS = ["mRNA", "CNV", "miRNA"]
BASE_K = {"mRNA": 200, "CNV": 50, "miRNA": 200}
MULTIPLIERS = [1, 2, 3, 4, 5]


def load_omics(omics):
    paths = {
        "mRNA": XENA / "HiSeqV2",
        "CNV": XENA / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "miRNA": XENA / "miRNA_HiSeq_gene",
    }
    return pd.read_csv(paths[omics], sep="\t", index_col=0, low_memory=False)


def align(omics):
    matrix = load_omics(omics)
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    cases = labels.loc[labels["pam50_4class"].notna() & (labels["pam50_4class"] != ""), "case_id"].tolist()
    case_cols = {}
    for col in matrix.columns:
        case_cols.setdefault(col[:12], []).append(col)
    best_cols = {c: ([x for x in cols if x.endswith("-01")] or cols)[0] for c, cols in case_cols.items()}
    aligned = [c for c in cases if c in best_cols]
    X = matrix[[best_cols[c] for c in aligned]].T.to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = labels.set_index("case_id").loc[aligned, "pam50_4class"].values
    return X, y


def make_selector(fs, k):
    var = VarianceThreshold(threshold=0.0)
    if fs == "V":
        return Pipeline([("var", var), ("f", SelectKBest(score_func=f_classif, k=k))])
    if fs == "V_F":
        return Pipeline([("var", var), ("f", SelectKBest(score_func=f_classif, k=k))])
    if fs == "V_L1":
        return Pipeline([("var", var), ("pre", SelectKBest(score_func=f_classif, k=min(k*5,2000))), ("l1", SelectFromModel(LinearSVC(penalty="l1", dual=False, C=0.1, max_iter=5000, random_state=RANDOM_STATE), max_features=k, threshold=-np.inf))])
    if fs == "V_NSRE":
        return Pipeline([("var", var), ("nsre", NSRESelector(prefilter_k=min(k*5,1000), k=k))])
    if fs == "V_F_L1":
        return Pipeline([("var", var), ("f", SelectKBest(score_func=f_classif, k=min(k*5,2000))), ("l1", SelectFromModel(LinearSVC(penalty="l1", dual=False, C=0.1, max_iter=5000, random_state=RANDOM_STATE), max_features=k, threshold=-np.inf))])
    if fs == "V_F_NSRE":
        return Pipeline([("var", var), ("f", SelectKBest(score_func=f_classif, k=min(k*5,1000))), ("nsre", NSRESelector(prefilter_k=min(k*5,1000), k=k))])
    if fs == "V_L1_F":
        return Pipeline([("var", var), ("l1", SelectFromModel(LinearSVC(penalty="l1", dual=False, C=0.1, max_iter=5000, random_state=RANDOM_STATE), max_features=min(k*5,2000), threshold=-np.inf)), ("f", SelectKBest(score_func=f_classif, k=k))])
    if fs == "V_L1_NSRE":
        return Pipeline([("var", var), ("l1", SelectFromModel(LinearSVC(penalty="l1", dual=False, C=0.1, max_iter=5000, random_state=RANDOM_STATE), max_features=min(k*5,2000), threshold=-np.inf)), ("nsre", NSRESelector(prefilter_k=min(k*5,1000), k=k))])
    if fs == "V_NSRE_F":
        return Pipeline([("var", var), ("nsre", NSRESelector(prefilter_k=min(k*5,1000), k=min(k*5,1000))), ("f", SelectKBest(score_func=f_classif, k=k))])
    if fs == "V_NSRE_L1":
        return Pipeline([("var", var), ("nsre", NSRESelector(prefilter_k=min(k*5,1000), k=min(k*5,1000))), ("l1", SelectFromModel(LinearSVC(penalty="l1", dual=False, C=0.1, max_iter=5000, random_state=RANDOM_STATE), max_features=k, threshold=-np.inf))])
    raise ValueError(fs)


FEATURE_METHODS = ["V", "V_F", "V_L1", "V_NSRE", "V_F_L1", "V_F_NSRE", "V_L1_F", "V_L1_NSRE", "V_NSRE_F", "V_NSRE_L1"]


def classifiers():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVC": SVC(kernel="linear", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def main():
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["omics","multiplier","k","feature_method","model","metric","mean","std"], delimiter="\t")
        writer.writeheader()
    for omics in OMICS:
        X, y = align(omics)
        for mult in MULTIPLIERS:
            k = BASE_K[omics] * mult
            for fs in FEATURE_METHODS:
                cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                for name, clf in classifiers().items():
                    accs, f1s = [], []
                    for tr, te in cv.split(np.zeros(len(y)), y):
                        sel = make_selector(fs, k); sel.fit(X[tr], y[tr])
                        Xtr_sel = sel.transform(X[tr]); Xte_sel = sel.transform(X[te])
                        scaler = StandardScaler().fit(Xtr_sel)
                        Xtr = scaler.transform(Xtr_sel); Xte = scaler.transform(Xte_sel)
                        clf.fit(Xtr, y[tr]); pred = clf.predict(Xte)
                        accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
                    with OUT.open("a", encoding="utf-8", newline="") as handle:
                        w = csv.DictWriter(handle, fieldnames=["omics","multiplier","k","feature_method","model","metric","mean","std"], delimiter="\t")
                        w.writerow({"omics":omics,"multiplier":mult,"k":k,"feature_method":fs,"model":name,"metric":"accuracy","mean":round(float(np.mean(accs)),4),"std":round(float(np.std(accs)),4)})
                        w.writerow({"omics":omics,"multiplier":mult,"k":k,"feature_method":fs,"model":name,"metric":"macro_f1","mean":round(float(np.mean(f1s)),4),"std":round(float(np.std(f1s)),4)})
                print(f"{omics} mult={mult} {fs} done")
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
