#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature-level early fusion and model-level late fusion."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "omics_integration_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
OMICS = ["mRNA", "CNV", "miRNA"]


def load_common(task):
    matrices = [pd.read_csv(FEAT_DIR / f"{o}_{task}_matrix.tsv", sep="\t", index_col=0) for o in OMICS]
    common = matrices[0].index
    for m in matrices[1:]:
        common = common.intersection(m.index)
    common = list(common)
    return [m.loc[common].to_numpy(dtype=float) for m in matrices], common


def base_models():
    return [
        ("mRNA", GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ("CNV", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ("miRNA", GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    ]


def pam50_late_fusion(blocks, y):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        probas = []
        for block, (name, clf) in zip(blocks, base_models()):
            scaler = StandardScaler().fit(block[tr])
            clf.fit(scaler.transform(block[tr]), y[tr])
            probas.append(clf.predict_proba(scaler.transform(block[te])))
        avg = np.mean(probas, axis=0)
        pred = avg.argmax(axis=1)
        accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s))


def pam50_early_fusion(blocks, y):
    X = np.hstack(blocks)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        model = Pipeline([("scale", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE))])
        model.fit(X[tr], y[tr]); pred = model.predict(X[te])
        accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s))


def os_late_fusion(blocks, y_event):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        probas = []
        for block, (name, clf) in zip(blocks, base_models()):
            scaler = StandardScaler().fit(block[tr])
            clf.fit(scaler.transform(block[tr]), y_event[tr])
            probas.append(clf.predict_proba(scaler.transform(block[te]))[:, 1])
        avg = np.mean(probas, axis=0)
        aucs.append(roc_auc_score(y_event[te], avg))
    return float(np.mean(aucs))


def os_early_fusion(blocks, y_event):
    X = np.hstack(blocks)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
        model.fit(X[tr], y_event[tr]); prob = model.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y_event[te], prob))
    return float(np.mean(aucs))


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    rows = []

    blocks, common = load_common("PAM50_4class")
    y = labels.set_index("case_id").loc[common, "pam50_4class"].values
    y = pd.factorize(y)[0]
    acc_late, f1_late = pam50_late_fusion(blocks, y)
    acc_early, f1_early = pam50_early_fusion(blocks, y)
    rows += [
        {"task": "PAM50", "method": "late_fusion_soft_vote", "metric": "accuracy", "value": round(acc_late, 4)},
        {"task": "PAM50", "method": "late_fusion_soft_vote", "metric": "macro_f1", "value": round(f1_late, 4)},
        {"task": "PAM50", "method": "early_fusion_concat", "metric": "accuracy", "value": round(acc_early, 4)},
        {"task": "PAM50", "method": "early_fusion_concat", "metric": "macro_f1", "value": round(f1_early, 4)},
    ]

    blocks, common = load_common("OS")
    y_event = labels.set_index("case_id").loc[common, "os_event"].astype(int).values
    auc_late = os_late_fusion(blocks, y_event)
    auc_early = os_early_fusion(blocks, y_event)
    rows += [
        {"task": "OS", "method": "late_fusion_soft_vote", "metric": "roc_auc", "value": round(auc_late, 4)},
        {"task": "OS", "method": "early_fusion_concat", "metric": "roc_auc", "value": round(auc_early, 4)},
    ]

    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "method", "metric", "value"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
