#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired tests for the main-text best-fusion comparisons.

The main manuscript uses one best fusion and one best single-omics model per
task/metric.  This script computes those comparisons on the same five folds and
records means, standard deviations, paired t-test, and Wilcoxon signed-rank
test so that the old Table 4 can be merged into Table 5.
"""

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

from run_advanced_method3_transformer import train as train_transformer  # noqa: E402
from run_comprehensive_batch1_single_omics import train_cnn, train_mlp  # noqa: E402
from run_fullsize_multimodal_integration import train_single_prob  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
SEL = DATA / "selected_features"
PAM_DIR = DATA / "final_datasets/PAM50"
SUR_DIR = DATA / "final_datasets/Survival"
IMG = DATA / "images"
OUT = DATA / "comprehensive_main_best_paired_tests.tsv"
OMICS = ["mRNA", "CNV", "miRNA"]
RANDOM_STATE = 42
K_FOLDS = 5
SIZE_PAM = {"mRNA": 20, "CNV": 8, "miRNA": 25}
SIZE_SUR = {"mRNA": 15, "CNV": 13, "miRNA": 25}


def load_pam50():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    case_lists = [
        pd.read_csv(SEL / f"{o}_PAM50_4class_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist()
        for o in OMICS
    ]
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
    return X, images, enc.fit_transform(y_raw), len(enc.classes_)


def load_survival():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    case_lists = [
        pd.read_csv(SEL / f"{o}_OS_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist()
        for o in OMICS
    ]
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
    return X, images, lab["os_event"].astype(int).values, lab["os_time_days"].astype(float).values


def pam50_metrics(prob, y):
    pred = prob.argmax(axis=1)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


def survival_metrics(prob, y_event, y_time):
    return roc_auc_score(y_event, prob), concordance_index(y_time, -prob, y_event)


def run_pam50():
    X, images, y, n_classes = load_pam50()
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    store = {
        "transformer": {"accuracy": [], "macro_f1": []},
        "mRNA_LR": {"accuracy": [], "macro_f1": []},
    }
    for tr, te in cv.split(np.zeros(len(y)), y):
        scalers = {o: StandardScaler().fit(X[o][tr]) for o in OMICS}
        Xtr = {o: scalers[o].transform(X[o][tr]) for o in OMICS}
        Xte = {o: scalers[o].transform(X[o][te]) for o in OMICS}

        lr = Pipeline([("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])
        lr.fit(Xtr["mRNA"], y[tr])
        p_lr = lr.predict_proba(Xte["mRNA"])
        acc, f1 = pam50_metrics(p_lr, y[te])
        store["mRNA_LR"]["accuracy"].append(acc)
        store["mRNA_LR"]["macro_f1"].append(f1)

        dims = {o: Xtr[o].shape[1] for o in OMICS}
        p_trans = train_transformer(Xtr, y[tr], Xte, dims, binary=False)
        acc, f1 = pam50_metrics(p_trans, y[te])
        store["transformer"]["accuracy"].append(acc)
        store["transformer"]["macro_f1"].append(f1)
    return store


def run_survival():
    X, images, y_event, y_time = load_survival()
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    store = {
        "late_fusion_avg": {"roc_auc": [], "c_index": []},
        "concat_mlp": {"roc_auc": [], "c_index": []},
        "mRNA_MLP": {"roc_auc": [], "c_index": []},
        "mRNA_FullSizeCNN": {"roc_auc": [], "c_index": []},
    }
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        scalers = {o: StandardScaler().fit(X[o][tr]) for o in OMICS}
        Xtr = {o: scalers[o].transform(X[o][tr]) for o in OMICS}
        Xte = {o: scalers[o].transform(X[o][te]) for o in OMICS}

        p_mlp = train_mlp(X["mRNA"][tr], y_event[tr], X["mRNA"][te], binary=True)
        auc, ci = survival_metrics(p_mlp, y_event[te], y_time[te])
        store["mRNA_MLP"]["roc_auc"].append(auc)
        store["mRNA_MLP"]["c_index"].append(ci)

        p_cnn = train_cnn(images["mRNA"][tr], y_event[tr], images["mRNA"][te], SIZE_SUR["mRNA"], binary=True)
        auc, ci = survival_metrics(p_cnn, y_event[te], y_time[te])
        store["mRNA_FullSizeCNN"]["roc_auc"].append(auc)
        store["mRNA_FullSizeCNN"]["c_index"].append(ci)

        probs = {}
        for o in OMICS:
            probs[o] = train_single_prob(images[o][tr], y_event[tr], images[o][te], SIZE_SUR[o], binary=True)
        late = (probs["mRNA"] + probs["CNV"] + probs["miRNA"]) / 3.0
        auc, ci = survival_metrics(late, y_event[te], y_time[te])
        store["late_fusion_avg"]["roc_auc"].append(auc)
        store["late_fusion_avg"]["c_index"].append(ci)

        Xcat_tr = np.hstack([X[o][tr] for o in OMICS])
        Xcat_te = np.hstack([X[o][te] for o in OMICS])
        p_concat = train_mlp(Xcat_tr, y_event[tr], Xcat_te, binary=True)
        auc, ci = survival_metrics(p_concat, y_event[te], y_time[te])
        store["concat_mlp"]["roc_auc"].append(auc)
        store["concat_mlp"]["c_index"].append(ci)
    return store


def summarize(task, metric, fusion_name, single_name, store, metric_label):
    a = np.array(store[fusion_name][metric_label], dtype=float)
    b = np.array(store[single_name][metric_label], dtype=float)
    t = ttest_rel(a, b)
    try:
        wp = float(wilcoxon(a, b, zero_method="wilcox", correction=False).pvalue)
    except Exception:
        wp = float("nan")
    return {
        "task": task,
        "metric": metric,
        "best_fusion": fusion_name,
        "best_single": single_name,
        "fusion_mean": round(float(np.mean(a)), 4),
        "fusion_std": round(float(np.std(a)), 4),
        "single_mean": round(float(np.mean(b)), 4),
        "single_std": round(float(np.std(b)), 4),
        "mean_diff": round(float(np.mean(a - b)), 4),
        "t_pvalue": round(float(t.pvalue), 4),
        "wilcoxon_pvalue": round(wp, 4),
    }


def main():
    pam = run_pam50()
    sur = run_survival()
    rows = [
        summarize("PAM50", "accuracy", "transformer", "mRNA_LR", pam, "accuracy"),
        summarize("PAM50", "macro_f1", "transformer", "mRNA_LR", pam, "macro_f1"),
        summarize("Survival", "roc_auc", "late_fusion_avg", "mRNA_MLP", sur, "roc_auc"),
        summarize("Survival", "c_index", "concat_mlp", "mRNA_FullSizeCNN", sur, "c_index"),
    ]
    fields = [
        "task",
        "metric",
        "best_fusion",
        "best_single",
        "fusion_mean",
        "fusion_std",
        "single_mean",
        "single_std",
        "mean_diff",
        "t_pvalue",
        "wilcoxon_pvalue",
    ]
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote -> {OUT}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
