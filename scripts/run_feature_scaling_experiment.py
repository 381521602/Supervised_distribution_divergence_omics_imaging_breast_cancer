#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature-count scaling experiment for PAM50."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from adaptive_jsd import JSDSelector  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
XENA = DATA / "external" / "xena"
OUT = DATA / "feature_scaling_experiment_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
OMICS = ["mRNA", "CNV", "miRNA"]
BASE_K = {"mRNA": 200, "CNV": 50, "miRNA": 200}
MULTIPLIERS = [1, 2, 3, 4, 5]
EPOCHS = 30
BATCH_SIZE = 64


def load_omics(omics):
    paths = {
        "mRNA": XENA / "HiSeqV2",
        "CNV": XENA / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "miRNA": XENA / "miRNA_HiSeq_gene",
    }
    return pd.read_csv(paths[omics], sep="\t", index_col=0, low_memory=False)


def align(omics, labels):
    matrix = load_omics(omics)
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


def make_selector(name, k):
    var = VarianceThreshold(threshold=0.0)
    if name == "FClassif":
        return Pipeline([("var", var), ("sel", SelectKBest(score_func=f_classif, k=k))])
    if name == "L1":
        return Pipeline([("var", var), ("pre", SelectKBest(score_func=f_classif, k=min(k * 5, 5000))), ("sel", SelectFromModel(LinearSVC(penalty="l1", dual=False, C=0.1, max_iter=5000, random_state=RANDOM_STATE), max_features=k, threshold=-np.inf))])
    if name == "JSD":
        return Pipeline([("var", var), ("sel", JSDSelector(prefilter_k=max(k * 5, 1000), k=k))])
    if name == "FClassif_JSD":
        return Pipeline([("var", var), ("f", SelectKBest(score_func=f_classif, k=min(k * 5, 5000))), ("jsd", JSDSelector(prefilter_k=min(k * 5, 5000), k=k))])
    raise ValueError(name)


def cv_lr(omics, k, fs):
    X, y = align(omics, pd.read_csv(LABELS, sep="\t", dtype={"case_id": str}))
    model = Pipeline([("sel", make_selector(fs, k)), ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1, error_score="raise")
    return float(np.mean(scores["test_accuracy"])), float(np.mean(scores["test_f1_macro"]))


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))
    def forward(self, x):
        return self.net(x)


def cv_mlp(omics, k):
    X, y = align(omics, pd.read_csv(LABELS, sep="\t", dtype={"case_id": str}))
    enc = LabelEncoder(); y = enc.fit_transform(y)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        sel = make_selector("FClassif_JSD", k)
        sel.fit(X[tr], y[tr])
        Xtr_sel = sel.transform(X[tr])
        Xte_sel = sel.transform(X[te])
        scaler = StandardScaler().fit(Xtr_sel)
        Xtr = scaler.transform(Xtr_sel)
        Xte = scaler.transform(Xte_sel)
        torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
        model = MLP(Xtr.shape[1], len(enc.classes_))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(); Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(y[tr], dtype=torch.long)
        n = Xt.shape[0]
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                if idx.shape[0] < 2: continue
                opt.zero_grad(); loss = crit(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte, dtype=torch.float32)).argmax(dim=1).numpy()
        accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s))


def main():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    rows = []
    for omics in OMICS:
        for mult in MULTIPLIERS:
            k = BASE_K[omics] * mult
            for fs in ["FClassif", "FClassif_JSD"]:
                acc, f1 = cv_lr(omics, k, fs)
                rows.append({"omics": omics, "k": k, "multiplier": mult, "feature_method": fs, "model": "LogisticRegression", "metric": "accuracy", "value": round(acc, 4)})
                rows.append({"omics": omics, "k": k, "multiplier": mult, "feature_method": fs, "model": "LogisticRegression", "metric": "macro_f1", "value": round(f1, 4)})
            acc, f1 = cv_mlp(omics, k)
            rows.append({"omics": omics, "k": k, "multiplier": mult, "feature_method": "FClassif_JSD", "model": "MLP", "metric": "accuracy", "value": round(acc, 4)})
            rows.append({"omics": omics, "k": k, "multiplier": mult, "feature_method": "FClassif_JSD", "model": "MLP", "metric": "macro_f1", "value": round(f1, 4)})
            print(f"{omics} k={k} done")
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["omics", "k", "multiplier", "feature_method", "model", "metric", "value"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
