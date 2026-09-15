#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FullSizeCNN 多组学整合：三分支拼接、门控融合、mRNA+miRNA、晚期概率融合。"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
SEL = DATA / "selected_features"
IMG = DATA / "images"
OUT = DATA / "fullsize_multimodal_integration_results.tsv"
PAIR_OUT = DATA / "fullsize_multimodal_integration_paired_tests.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
OMICS = ["mRNA", "CNV", "miRNA"]


class Branch(nn.Module):
    def __init__(self, size, out_ch=32):
        super().__init__()
        self.conv = nn.Conv2d(1, out_ch, kernel_size=(size, size))

    def forward(self, x):
        return F.relu(self.conv(x)).flatten(1)


class TripleConcatNet(nn.Module):
    def __init__(self, sizes, out_dim):
        super().__init__()
        self.branches = nn.ModuleDict({o: Branch(sizes[o]) for o in OMICS})
        self.head = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, xs):
        return self.head(torch.cat([self.branches[o](xs[o]) for o in OMICS], dim=1))


class TripleGatedNet(nn.Module):
    def __init__(self, sizes, out_dim):
        super().__init__()
        self.branches = nn.ModuleDict({o: Branch(sizes[o]) for o in OMICS})
        self.gate = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 3), nn.Softmax(dim=1))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, xs):
        feats = [self.branches[o](xs[o]) for o in OMICS]
        stacked = torch.stack(feats, dim=1)  # B,3,32
        weights = self.gate(torch.cat(feats, dim=1)).unsqueeze(-1)
        z = (stacked * weights).sum(dim=1)
        return self.head(z)


class TwoBranchConcatNet(nn.Module):
    def __init__(self, sizes, out_dim):
        super().__init__()
        self.branches = nn.ModuleDict({"mRNA": Branch(sizes["mRNA"]), "miRNA": Branch(sizes["miRNA"])})
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, xs):
        return self.head(torch.cat([self.branches["mRNA"](xs["mRNA"]), self.branches["miRNA"](xs["miRNA"])], dim=1))


def train_multi(model, Xtr, ytr, Xte, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = {k: torch.tensor(v, dtype=torch.float32) for k, v in Xtr.items()}
    if binary:
        crit = nn.BCEWithLogitsLoss()
        yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    else:
        crit = nn.CrossEntropyLoss()
        yt = torch.tensor(ytr, dtype=torch.long)
    n = yt.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad()
            loss = crit(model({k: v[idx] for k, v in Xt.items()}), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model({k: torch.tensor(v, dtype=torch.float32) for k, v in Xte.items()})
    if binary:
        return torch.sigmoid(out).numpy().ravel()
    return F.softmax(out, dim=1).numpy()


def train_single_prob(Xtr, ytr, Xte, size, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(size, size)),
        nn.Flatten(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1 if binary else 4),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    if binary:
        crit = nn.BCEWithLogitsLoss()
        yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    else:
        crit = nn.CrossEntropyLoss()
        yt = torch.tensor(ytr, dtype=torch.long)
    n = yt.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte, dtype=torch.float32))
    if binary:
        return torch.sigmoid(out).numpy().ravel()
    return F.softmax(out, dim=1).numpy()


def load_pam50():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    case_lists = [
        pd.read_csv(SEL / f"{o}_PAM50_4class_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist()
        for o in OMICS
    ]
    common = [c for c in case_lists[0] if c in set(case_lists[1]) and c in set(case_lists[2])]
    sizes = {"mRNA": 20, "CNV": 8, "miRNA": 25}
    X = {}
    for o, cases in zip(OMICS, case_lists):
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X[o] = np.load(IMG / "PAM50" / o / "images.npy")[idx]
    y_raw = labels.set_index("case_id").loc[common, "pam50_4class"].values
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    return X, y, sizes, len(enc.classes_), common


def load_survival():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    case_lists = [
        pd.read_csv(SEL / f"{o}_OS_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist()
        for o in OMICS
    ]
    common = [c for c in case_lists[0] if c in set(case_lists[1]) and c in set(case_lists[2])]
    sizes = {"mRNA": 15, "CNV": 13, "miRNA": 25}
    X = {}
    for o, cases in zip(OMICS, case_lists):
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X[o] = np.load(IMG / "Survival" / o / "images.npy")[idx]
    lab = labels.set_index("case_id").loc[common]
    y_event = lab["os_event"].astype(int).values
    y_time = lab["os_time_days"].astype(float).values
    return X, y_event, y_time, sizes, common


def empty_fold_store(task):
    schemes = ["single_mRNA", "single_CNV", "single_miRNA", "triple_concat", "triple_gated", "mrna_mirna_concat", "late_fusion_avg"]
    metrics = ["accuracy", "macro_f1"] if task == "PAM50" else ["roc_auc", "c_index"]
    return {s: {m: [] for m in metrics} for s in schemes}


def add_pam50_metric(store, scheme, pred_probs, yte):
    pred = pred_probs.argmax(axis=1)
    store[scheme]["accuracy"].append(accuracy_score(yte, pred))
    store[scheme]["macro_f1"].append(f1_score(yte, pred, average="macro"))


def add_survival_metric(store, scheme, prob, yte, y_time_te):
    store[scheme]["roc_auc"].append(roc_auc_score(yte, prob))
    store[scheme]["c_index"].append(concordance_index(y_time_te, -prob, yte))


def run_pam50():
    X, y, sizes, n_classes, _ = load_pam50()
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    store = empty_fold_store("PAM50")
    for tr, te in cv.split(np.zeros(len(y)), y):
        Xtr = {o: v[tr] for o, v in X.items()}
        Xte = {o: v[te] for o, v in X.items()}
        ytr, yte = y[tr], y[te]

        probs = {}
        for o in OMICS:
            probs[o] = train_single_prob(Xtr[o], ytr, Xte[o], sizes[o], binary=False)
            add_pam50_metric(store, f"single_{o}", probs[o], yte)

        model = TripleConcatNet(sizes, n_classes)
        p = train_multi(model, Xtr, ytr, Xte, binary=False)
        add_pam50_metric(store, "triple_concat", p, yte)

        model = TripleGatedNet(sizes, n_classes)
        p = train_multi(model, Xtr, ytr, Xte, binary=False)
        add_pam50_metric(store, "triple_gated", p, yte)

        model = TwoBranchConcatNet(sizes, n_classes)
        p = train_multi(model, Xtr, ytr, Xte, binary=False)
        add_pam50_metric(store, "mrna_mirna_concat", p, yte)

        late = (probs["mRNA"] + probs["CNV"] + probs["miRNA"]) / 3.0
        add_pam50_metric(store, "late_fusion_avg", late, yte)
    return store


def run_survival():
    X, y_event, y_time, sizes, _ = load_survival()
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    store = empty_fold_store("Survival")
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        Xtr = {o: v[tr] for o, v in X.items()}
        Xte = {o: v[te] for o, v in X.items()}
        ytr, yte = y_event[tr], y_event[te]
        y_time_te = y_time[te]

        probs = {}
        for o in OMICS:
            probs[o] = train_single_prob(Xtr[o], ytr, Xte[o], sizes[o], binary=True)
            add_survival_metric(store, f"single_{o}", probs[o], yte, y_time_te)

        model = TripleConcatNet(sizes, 1)
        p = train_multi(model, Xtr, ytr, Xte, binary=True)
        add_survival_metric(store, "triple_concat", p, yte, y_time_te)

        model = TripleGatedNet(sizes, 1)
        p = train_multi(model, Xtr, ytr, Xte, binary=True)
        add_survival_metric(store, "triple_gated", p, yte, y_time_te)

        model = TwoBranchConcatNet(sizes, 1)
        p = train_multi(model, Xtr, ytr, Xte, binary=True)
        add_survival_metric(store, "mrna_mirna_concat", p, yte, y_time_te)

        late = (probs["mRNA"] + probs["CNV"] + probs["miRNA"]) / 3.0
        add_survival_metric(store, "late_fusion_avg", late, yte, y_time_te)
    return store


def write_rows(task, store):
    metrics = ["accuracy", "macro_f1"] if task == "PAM50" else ["roc_auc", "c_index"]
    rows = []
    for scheme, m in store.items():
        for metric in metrics:
            rows.append({"scheme": scheme, "task": task, "metric": metric, "mean": round(float(np.mean(m[metric])), 4), "std": round(float(np.std(m[metric])), 4)})
    return rows


def paired_tests(task, store):
    if task == "PAM50":
        metrics = ["accuracy", "macro_f1"]
        single_names = ["single_mRNA", "single_CNV", "single_miRNA"]
        integ_names = ["triple_concat", "triple_gated", "mrna_mirna_concat", "late_fusion_avg"]
    else:
        metrics = ["roc_auc", "c_index"]
        single_names = ["single_mRNA", "single_CNV", "single_miRNA"]
        integ_names = ["triple_concat", "triple_gated", "mrna_mirna_concat", "late_fusion_avg"]
    rows = []
    for metric in metrics:
        best_integ = max(integ_names, key=lambda s: np.mean(store[s][metric]))
        best_single = max(single_names, key=lambda s: np.mean(store[s][metric]))
        a = np.array(store[best_integ][metric], dtype=float)
        b = np.array(store[best_single][metric], dtype=float)
        t = ttest_rel(a, b)
        try:
            wp = float(wilcoxon(a, b, zero_method="wilcox", correction=False).pvalue)
        except Exception:
            wp = float("nan")
        rows.append({
            "task": task,
            "metric": metric,
            "best_integration": best_integ,
            "best_single": best_single,
            "mean_diff_integ_minus_single": round(float(np.mean(a - b)), 4),
            "t_pvalue": round(float(t.pvalue), 4),
            "wilcoxon_pvalue": round(wp, 4),
        })
    return rows


def main():
    pam = run_pam50()
    sur = run_survival()
    rows = write_rows("PAM50", pam) + write_rows("Survival", sur)
    pair_rows = paired_tests("PAM50", pam) + paired_tests("Survival", sur)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scheme", "task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    with PAIR_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "metric", "best_integration", "best_single", "mean_diff_integ_minus_single", "t_pvalue", "wilcoxon_pvalue"], delimiter="\t")
        w.writeheader()
        w.writerows(pair_rows)
    print(f"Wrote -> {OUT}")
    print(f"Wrote -> {PAIR_OUT}")
    for r in rows:
        print(r["scheme"], r["task"], r["metric"], r["mean"], "+-", r["std"])
    for r in pair_rows:
        print(r)


if __name__ == "__main__":
    main()
