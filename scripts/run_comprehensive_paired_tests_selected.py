#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selected best-model paired tests on the three-omics intersection."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_comprehensive_batch1_single_omics import train_mlp  # noqa: E402
from run_fullsize_multimodal_integration import train_single_prob  # noqa: E402
from run_advanced_method3_transformer import TransformerFusion, train as train_transformer  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
SEL = DATA / "selected_features"
PAM_DIR = DATA / "final_datasets/PAM50"
SUR_DIR = DATA / "final_datasets/Survival"
IMG = DATA / "images"
OUT = DATA / "comprehensive_paired_tests_selected.tsv"
OMICS = ["mRNA", "CNV", "miRNA"]
RANDOM_STATE = 42
K_FOLDS = 5
SIZE_PAM = {"mRNA": 20, "CNV": 8, "miRNA": 25}
SIZE_SUR = {"mRNA": 15, "CNV": 13, "miRNA": 25}


def load_pam50():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    lists = [pd.read_csv(SEL / f"{o}_PAM50_4class_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in OMICS]
    common = [c for c in lists[0] if c in set(lists[1]) and c in set(lists[2])]
    X, images = {}, {}
    for o, cases in zip(OMICS, lists):
        df = pd.read_csv(PAM_DIR / f"{o}_PAM50_final.tsv", sep="\t")
        Xmat = df.drop(columns=["pam50"]).to_numpy(dtype=float)
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X[o] = Xmat[idx]
        images[o] = np.load(IMG / "PAM50" / o / "images.npy")[idx]
    y_raw = labels.set_index("case_id").loc[common, "pam50_4class"].values
    enc = LabelEncoder()
    return X, images, enc.fit_transform(y_raw), len(enc.classes_)


def load_survival():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    lists = [pd.read_csv(SEL / f"{o}_OS_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in OMICS]
    common = [c for c in lists[0] if c in set(lists[1]) and c in set(lists[2])]
    X, images = {}, {}
    for o, cases in zip(OMICS, lists):
        df = pd.read_csv(SUR_DIR / f"{o}_Survival_final.tsv", sep="\t")
        Xmat = df.drop(columns=["case_id", "os_event", "os_time_days"]).to_numpy(dtype=float)
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X[o] = Xmat[idx]
        images[o] = np.load(IMG / "Survival" / o / "images.npy")[idx]
    lab = labels.set_index("case_id").loc[common]
    return X, images, lab["os_event"].astype(int).values, lab["os_time_days"].astype(float).values


def ml_predict(Xtr, ytr, Xte, binary):
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1] if binary else pipe.predict_proba(Xte)


def add_metric(d, name, task, p, yte, y_time_te=None):
    if task == "PAM50":
        pred = p.argmax(axis=1) if p.ndim == 2 else p
        d[name]["accuracy"].append(accuracy_score(yte, pred))
        d[name]["macro_f1"].append(f1_score(yte, pred, average="macro"))
    else:
        d[name]["roc_auc"].append(roc_auc_score(yte, p))
        d[name]["c_index"].append(concordance_index(y_time_te, -p, yte))


def run_pam50(X, images, y, sizes):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    d = {name: {"accuracy": [], "macro_f1": []} for name in ["single_mRNA_ML", "within_mRNA_avg", "comprehensive_transformer"]}
    for tr, te in cv.split(np.zeros(len(y)), y):
        scalers = {o: StandardScaler().fit(X[o][tr]) for o in OMICS}
        Xtr = {o: scalers[o].transform(X[o][tr]) for o in OMICS}
        Xte = {o: scalers[o].transform(X[o][te]) for o in OMICS}

        p_ml = ml_predict(Xtr["mRNA"], y[tr], Xte["mRNA"], binary=False)
        p_mlp = train_mlp(Xtr["mRNA"], y[tr], Xte["mRNA"], binary=False)
        p_cnn = train_single_prob(images["mRNA"][tr], y[tr], images["mRNA"][te], sizes["mRNA"], binary=False)
        avg = (p_ml + p_mlp + p_cnn) / 3.0
        add_metric(d, "single_mRNA_ML", "PAM50", p_ml, y[te])
        add_metric(d, "within_mRNA_avg", "PAM50", avg, y[te])

        dims = {o: Xtr[o].shape[1] for o in OMICS}
        p_trans = train_transformer(Xtr, y[tr], Xte, dims, binary=False)
        add_metric(d, "comprehensive_transformer", "PAM50", p_trans, y[te])
    return d


def run_survival(X, images, y_event, y_time, sizes):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    d = {name: {"roc_auc": [], "c_index": []} for name in ["single_mRNA_MLP", "within_mRNA_avg", "comprehensive_concat_mlp"]}
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        scalers = {o: StandardScaler().fit(X[o][tr]) for o in OMICS}
        Xtr = {o: scalers[o].transform(X[o][tr]) for o in OMICS}
        Xte = {o: scalers[o].transform(X[o][te]) for o in OMICS}

        p_ml = ml_predict(Xtr["mRNA"], y_event[tr], Xte["mRNA"], binary=True)
        p_mlp = train_mlp(Xtr["mRNA"], y_event[tr], Xte["mRNA"], binary=True)
        p_cnn = train_single_prob(images["mRNA"][tr], y_event[tr], images["mRNA"][te], sizes["mRNA"], binary=True)
        avg = (p_ml + p_mlp + p_cnn) / 3.0
        add_metric(d, "single_mRNA_MLP", "Survival", p_mlp, y_event[te], y_time[te])
        add_metric(d, "within_mRNA_avg", "Survival", avg, y_event[te], y_time[te])

        Xcat_tr = np.hstack([Xtr[o] for o in OMICS])
        Xcat_te = np.hstack([Xte[o] for o in OMICS])
        p_cat = train_mlp(Xcat_tr, y_event[tr], Xcat_te, binary=True)
        add_metric(d, "comprehensive_concat_mlp", "Survival", p_cat, y_event[te], y_time[te])
    return d


def paired(task, metric, d, comp_name):
    rows = []
    for other in d:
        if other == comp_name:
            continue
        a = np.array(d[comp_name][metric], dtype=float)
        b = np.array(d[other][metric], dtype=float)
        t = ttest_rel(a, b)
        try:
            wp = float(wilcoxon(a, b, zero_method="wilcox", correction=False).pvalue)
        except Exception:
            wp = float("nan")
        rows.append({"task": task, "metric": metric, "comparison": f"{comp_name} vs {other}", "mean_diff": round(float(np.mean(a - b)), 4), "t_pvalue": round(float(t.pvalue), 4), "wilcoxon_pvalue": round(wp, 4)})
    return rows


def main():
    X, images, y, sizes = load_pam50()
    sizes = {o: SIZE_PAM[o] for o in OMICS}
    pam = run_pam50(X, images, y, sizes)
    X, images, y_event, y_time = load_survival()
    sizes = {o: SIZE_SUR[o] for o in OMICS}
    sur = run_survival(X, images, y_event, y_time, sizes)

    rows = []
    rows += paired("PAM50", "accuracy", pam, "comprehensive_transformer")
    rows += paired("PAM50", "macro_f1", pam, "comprehensive_transformer")
    rows += paired("Survival", "roc_auc", sur, "comprehensive_concat_mlp")
    rows += paired("Survival", "c_index", sur, "comprehensive_concat_mlp")
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "metric", "comparison", "mean_diff", "t_pvalue", "wilcoxon_pvalue"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
