#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pairwise-omics stacking on pair-specific intersections."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_comprehensive_batch1_single_omics import train_mlp  # noqa: E402
from run_fullsize_multimodal_integration import train_single_prob  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
SEL = DATA / "selected_features"
PAM_DIR = DATA / "final_datasets/PAM50"
SUR_DIR = DATA / "final_datasets/Survival"
IMG = DATA / "images"
OUT = DATA / "pairwise_stacking_results.tsv"
PAIRS = [("mRNA", "CNV"), ("mRNA", "miRNA"), ("CNV", "miRNA")]
RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
SIZE_PAM = {"mRNA": 20, "CNV": 8, "miRNA": 25}
SIZE_SUR = {"mRNA": 15, "CNV": 13, "miRNA": 25}


def load_pam50(pair):
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    lists = [pd.read_csv(SEL / f"{o}_PAM50_4class_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in pair]
    common = [c for c in lists[0] if c in set(lists[1])]
    X, images = {}, {}
    for o, cases in zip(pair, lists):
        df = pd.read_csv(PAM_DIR / f"{o}_PAM50_final.tsv", sep="\t")
        Xmat = df.drop(columns=["pam50"]).to_numpy(dtype=float)
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X[o] = Xmat[idx]
        images[o] = np.load(IMG / "PAM50" / o / "images.npy")[idx]
    y_raw = labels.set_index("case_id").loc[common, "pam50_4class"].values
    enc = LabelEncoder()
    return X, images, enc.fit_transform(y_raw), len(enc.classes_)


def load_survival(pair):
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    lists = [pd.read_csv(SEL / f"{o}_OS_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in pair]
    common = [c for c in lists[0] if c in set(lists[1])]
    X, images = {}, {}
    for o, cases in zip(pair, lists):
        df = pd.read_csv(SUR_DIR / f"{o}_Survival_final.tsv", sep="\t")
        Xmat = df.drop(columns=["case_id", "os_event", "os_time_days"]).to_numpy(dtype=float)
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X[o] = Xmat[idx]
        images[o] = np.load(IMG / "Survival" / o / "images.npy")[idx]
    lab = labels.set_index("case_id").loc[common]
    return X, images, lab["os_event"].astype(int).values, lab["os_time_days"].astype(float).values


def base_predict(kind, o, Xtr_o, ytr, Xte_o, imgs_tr, imgs_te, size, binary):
    if kind == "ML":
        pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])
        pipe.fit(Xtr_o, ytr)
        return pipe.predict_proba(Xte_o)[:, 1] if binary else pipe.predict_proba(Xte_o)
    if kind == "MLP":
        return train_mlp(Xtr_o, ytr, Xte_o, binary=binary)
    if kind == "CNN":
        return train_single_prob(imgs_tr, ytr, imgs_te, size, binary=binary)
    raise ValueError(kind)


def make_oof(kind, o, X_all, images_all, outer_idx, y_all, size, binary, n_classes):
    n = len(outer_idx)
    oof = np.zeros(n) if binary else np.zeros((n, n_classes), dtype=float)
    y_tr = y_all[outer_idx]
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for itr, ite in inner.split(np.zeros(n), y_tr):
        oof[ite] = base_predict(kind, o, X_all[o][outer_idx[itr]], y_tr[itr], X_all[o][outer_idx[ite]], images_all[o][outer_idx[itr]], images_all[o][outer_idx[ite]], size, binary)
    return oof


def test_feature(kind, o, X_all, images_all, train_idx, test_idx, y_all, size, binary):
    return base_predict(kind, o, X_all[o][train_idx], y_all[train_idx], X_all[o][test_idx], images_all[o][train_idx], images_all[o][test_idx], size, binary)


def eval_pair(pair, X, images, y, sizes, binary, y_time=None):
    n_classes = 1 if binary else 4
    base_names = [f"{o}_{k}" for o in pair for k in ["ML", "MLP", "CNN"]]
    cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s, aucs, cis = [], [], [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        oofs, tests = [], []
        for o in pair:
            for k in ["ML", "MLP", "CNN"]:
                oofs.append(make_oof(k, o, X, images, tr, y, sizes[o], binary, n_classes))
                tests.append(test_feature(k, o, X, images, tr, te, y, sizes[o], binary))
        if binary:
            Xtr_meta = np.column_stack(oofs)
            Xte_meta = np.column_stack(tests)
            meta = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])
            meta.fit(Xtr_meta, y[tr])
            p = meta.predict_proba(Xte_meta)[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            cis.append(concordance_index(y_time[te], -p, y[te]))
        else:
            Xtr_meta = np.hstack(oofs)
            Xte_meta = np.hstack(tests)
            meta = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])
            meta.fit(Xtr_meta, y[tr])
            pred = meta.predict(Xte_meta)
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="macro"))
    rows = []
    if binary:
        rows.append({"pair": "+".join(pair), "task": "Survival", "metric": "roc_auc", "mean": round(float(np.mean(aucs)), 4), "std": round(float(np.std(aucs)), 4)})
        rows.append({"pair": "+".join(pair), "task": "Survival", "metric": "c_index", "mean": round(float(np.mean(cis)), 4), "std": round(float(np.std(cis)), 4)})
    else:
        rows.append({"pair": "+".join(pair), "task": "PAM50", "metric": "accuracy", "mean": round(float(np.mean(accs)), 4), "std": round(float(np.std(accs)), 4)})
        rows.append({"pair": "+".join(pair), "task": "PAM50", "metric": "macro_f1", "mean": round(float(np.mean(f1s)), 4), "std": round(float(np.std(f1s)), 4)})
    return rows


def main():
    rows = []
    for pair in PAIRS:
        X, images, y, n_classes = load_pam50(pair)
        rows += eval_pair(pair, X, images, y, {o: SIZE_PAM[o] for o in pair}, binary=False)
        print(f"done PAM50 {pair}", flush=True)
    for pair in PAIRS:
        X, images, y_event, y_time = load_survival(pair)
        rows += eval_pair(pair, X, images, y_event, {o: SIZE_SUR[o] for o in pair}, binary=True, y_time=y_time)
        print(f"done Survival {pair}", flush=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
