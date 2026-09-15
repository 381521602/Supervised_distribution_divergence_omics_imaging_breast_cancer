#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final 9-model stacking on the three-omics intersection."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lifelines.utils import concordance_index


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
OUT = DATA / "final_9model_stacking_results.tsv"
RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
OMICS = ["mRNA", "CNV", "miRNA"]
SIZE_PAM = {"mRNA": 20, "CNV": 8, "miRNA": 25}
SIZE_SUR = {"mRNA": 15, "CNV": 13, "miRNA": 25}


def load_pam50():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    case_lists = [pd.read_csv(SEL / f"{o}_PAM50_4class_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in OMICS]
    common = [c for c in case_lists[0] if c in set(case_lists[1]) and c in set(case_lists[2])]
    X, images = {}, {}
    for o, cases in zip(OMICS, case_lists):
        df = pd.read_csv(PAM_DIR / f"{o}_PAM50_final.tsv", sep="\t")
        Xmat = df.drop(columns=["pam50"]).to_numpy(dtype=float)
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X[o] = Xmat[idx]
        images[o] = np.load(IMG / "PAM50" / o / "images.npy")[idx]
    y_raw = labels.set_index("case_id").loc[common, "pam50_4class"].values
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    return X, images, y, len(enc.classes_)


def load_survival():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    case_lists = [pd.read_csv(SEL / f"{o}_OS_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in OMICS]
    common = [c for c in case_lists[0] if c in set(case_lists[1]) and c in set(case_lists[2])]
    X, images = {}, {}
    for o, cases in zip(OMICS, case_lists):
        df = pd.read_csv(SUR_DIR / f"{o}_Survival_final.tsv", sep="\t")
        Xmat = df.drop(columns=["case_id", "os_event", "os_time_days"]).to_numpy(dtype=float)
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X[o] = Xmat[idx]
        images[o] = np.load(IMG / "Survival" / o / "images.npy")[idx]
    lab = labels.set_index("case_id").loc[common]
    y_event = lab["os_event"].astype(int).values
    y_time = lab["os_time_days"].astype(float).values
    return X, images, y_event, y_time


def base_predict(kind, omics, Xtr_omics, ytr, Xte_omics, imgs_tr, imgs_te, size, binary, n_classes):
    if kind == "ML":
        pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])
        pipe.fit(Xtr_omics, ytr)
        if binary:
            return pipe.predict_proba(Xte_omics)[:, 1]
        return pipe.predict_proba(Xte_omics)
    if kind == "MLP":
        return train_mlp(Xtr_omics, ytr, Xte_omics, binary=binary)
    if kind == "CNN":
        return train_single_prob(imgs_tr, ytr, imgs_te, size, binary=binary)
    raise ValueError(kind)


def make_oof(kind, omics, X_all, images_all, outer_idx, y_all, size, binary, n_classes):
    n = len(outer_idx)
    if binary:
        oof = np.zeros(n, dtype=float)
    else:
        oof = np.zeros((n, n_classes), dtype=float)
    y_tr = y_all[outer_idx]
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for itr, ite in inner_cv.split(np.zeros(n), y_tr):
        Xtr_omics = X_all[omics][outer_idx[itr]]
        Xte_omics = X_all[omics][outer_idx[ite]]
        imgs_tr = images_all[omics][outer_idx[itr]]
        imgs_te = images_all[omics][outer_idx[ite]]
        oof[ite] = base_predict(kind, omics, Xtr_omics, y_tr[itr], Xte_omics, imgs_tr, imgs_te, size, binary, n_classes)
    return oof


def test_features(kind, omics, X_all, images_all, train_idx, test_idx, y_all, size, binary, n_classes):
    Xtr_omics = X_all[omics][train_idx]
    Xte_omics = X_all[omics][test_idx]
    imgs_tr = images_all[omics][train_idx]
    imgs_te = images_all[omics][test_idx]
    return base_predict(kind, omics, Xtr_omics, y_all[train_idx], Xte_omics, imgs_tr, imgs_te, size, binary, n_classes)


def eval_stacking(X, images, y, sizes, binary, y_time=None):
    n_classes = 1 if binary else 4
    base_names = [f"{o}_{kind}" for o in OMICS for kind in ["ML", "MLP", "CNN"]]
    cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metrics = {name: {"accuracy": [], "macro_f1": [], "roc_auc": [], "c_index": []} for name in base_names}
    metrics["Stacking_9"] = {"accuracy": [], "macro_f1": [], "roc_auc": [], "c_index": []}

    for tr, te in cv.split(np.zeros(len(y)), y):
        base_test_feats = {}
        base_oof_feats = {}
        for name, (o, kind) in zip(base_names, [(o, k) for o in OMICS for k in ["ML", "MLP", "CNN"]]):
            base_oof_feats[name] = make_oof(kind, o, X, images, tr, y, sizes[o], binary, n_classes)
            base_test_feats[name] = test_features(kind, o, X, images, tr, te, y, sizes[o], binary, n_classes)

        if binary:
            Xtr_meta = np.column_stack([base_oof_feats[n] for n in base_names])
            Xte_meta = np.column_stack([base_test_feats[n] for n in base_names])
            meta = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])
            meta.fit(Xtr_meta, y[tr])
            stacking_prob = meta.predict_proba(Xte_meta)[:, 1]
            for name in base_names:
                prob = base_test_feats[name]
                metrics[name]["roc_auc"].append(roc_auc_score(y[te], prob))
                metrics[name]["c_index"].append(concordance_index(y_time[te], -prob, y[te]))
            metrics["Stacking_9"]["roc_auc"].append(roc_auc_score(y[te], stacking_prob))
            metrics["Stacking_9"]["c_index"].append(concordance_index(y_time[te], -stacking_prob, y[te]))
        else:
            Xtr_meta = np.hstack([base_oof_feats[n] for n in base_names])
            Xte_meta = np.hstack([base_test_feats[n] for n in base_names])
            meta = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])
            meta.fit(Xtr_meta, y[tr])
            stacking_prob = meta.predict_proba(Xte_meta)
            stacking_pred = stacking_prob.argmax(axis=1)
            for name in base_names:
                pred = base_test_feats[name].argmax(axis=1)
                metrics[name]["accuracy"].append(accuracy_score(y[te], pred))
                metrics[name]["macro_f1"].append(f1_score(y[te], pred, average="macro"))
            metrics["Stacking_9"]["accuracy"].append(accuracy_score(y[te], stacking_pred))
            metrics["Stacking_9"]["macro_f1"].append(f1_score(y[te], stacking_pred, average="macro"))

    rows = []
    for name, m in metrics.items():
        if binary:
            rows.append({"model": name, "task": "Survival", "metric": "roc_auc", "mean": round(float(np.mean(m["roc_auc"])), 4), "std": round(float(np.std(m["roc_auc"])), 4)})
            rows.append({"model": name, "task": "Survival", "metric": "c_index", "mean": round(float(np.mean(m["c_index"])), 4), "std": round(float(np.std(m["c_index"])), 4)})
        else:
            rows.append({"model": name, "task": "PAM50", "metric": "accuracy", "mean": round(float(np.mean(m["accuracy"])), 4), "std": round(float(np.std(m["accuracy"])), 4)})
            rows.append({"model": name, "task": "PAM50", "metric": "macro_f1", "mean": round(float(np.mean(m["macro_f1"])), 4), "std": round(float(np.std(m["macro_f1"])), 4)})
    return rows


def main():
    X, images, y, n_classes = load_pam50()
    rows = eval_stacking(X, images, y, {o: SIZE_PAM[o] for o in OMICS}, binary=False)
    print("done PAM50", flush=True)
    X, images, y_event, y_time = load_survival()
    rows += eval_stacking(X, images, y_event, {o: SIZE_SUR[o] for o in OMICS}, binary=True, y_time=y_time)
    print("done Survival", flush=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
