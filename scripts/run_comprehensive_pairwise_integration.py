#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive pairwise-omics integration on pair-specific intersections."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_comprehensive_batch1_single_omics import ml_models, train_mlp, risk_score  # noqa: E402
from run_fullsize_multimodal_integration import Branch, train_multi, train_single_prob  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
SEL = DATA / "selected_features"
PAM_DIR = DATA / "final_datasets/PAM50"
SUR_DIR = DATA / "final_datasets/Survival"
IMG = DATA / "images"
OUT = DATA / "comprehensive_pairwise_integration_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
PAIRS = [("mRNA", "CNV"), ("mRNA", "miRNA"), ("CNV", "miRNA")]
SIZE_PAM = {"mRNA": 20, "CNV": 8, "miRNA": 25}
SIZE_SUR = {"mRNA": 15, "CNV": 13, "miRNA": 25}


class TwoBranchConcatNet(nn.Module):
    def __init__(self, sizes, pair, out_dim):
        super().__init__()
        self.pair = pair
        self.branches = nn.ModuleDict({o: Branch(sizes[o]) for o in pair})
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, xs):
        return self.head(torch.cat([self.branches[o](xs[o]) for o in self.pair], dim=1))


class TwoBranchGatedNet(nn.Module):
    def __init__(self, sizes, pair, out_dim):
        super().__init__()
        self.pair = pair
        self.branches = nn.ModuleDict({o: Branch(sizes[o]) for o in pair})
        self.gate = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2), nn.Softmax(dim=1))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, xs):
        feats = [self.branches[o](xs[o]) for o in self.pair]
        stacked = torch.stack(feats, dim=1)
        weights = self.gate(torch.cat(feats, dim=1)).unsqueeze(-1)
        z = (stacked * weights).sum(dim=1)
        return self.head(z)


def load_cases(task, pair):
    suffix = "PAM50_4class_matrix.tsv" if task == "PAM50" else "OS_matrix.tsv"
    lists = [pd.read_csv(SEL / f"{o}_{suffix}", sep="\t", usecols=[0], index_col=0).index.tolist() for o in pair]
    return [c for c in lists[0] if c in set(lists[1])], lists


def load_pam50(pair):
    common, lists = load_cases("PAM50", pair)
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
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
    y = enc.fit_transform(y_raw)
    return X, images, y, len(enc.classes_)


def load_survival(pair):
    common, lists = load_cases("Survival", pair)
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    X, images = {}, {}
    for o, cases in zip(pair, lists):
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


def eval_pam50(pair, X, images, y, n_classes):
    sizes = {o: SIZE_PAM[o] for o in pair}
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    store = {}
    for name in ml_models():
        store[f"Concat_{name}"] = {"accuracy": [], "macro_f1": []}
    store["Concat_MLP"] = {"accuracy": [], "macro_f1": []}
    store["CNN_TwoConcat"] = {"accuracy": [], "macro_f1": []}
    store["CNN_TwoGated"] = {"accuracy": [], "macro_f1": []}
    store["CNN_LateAvg"] = {"accuracy": [], "macro_f1": []}
    for tr, te in cv.split(np.zeros(len(y)), y):
        Xtr = np.hstack([X[o][tr] for o in pair])
        Xte = np.hstack([X[o][te] for o in pair])
        for name, clf in ml_models().items():
            pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
            pipe.fit(Xtr, y[tr])
            pred = pipe.predict(Xte)
            store[f"Concat_{name}"]["accuracy"].append(accuracy_score(y[te], pred))
            store[f"Concat_{name}"]["macro_f1"].append(f1_score(y[te], pred, average="macro"))
        p = train_mlp(Xtr, y[tr], Xte, binary=False)
        pred = p.argmax(axis=1)
        store["Concat_MLP"]["accuracy"].append(accuracy_score(y[te], pred))
        store["Concat_MLP"]["macro_f1"].append(f1_score(y[te], pred, average="macro"))

        Xtr_img = {o: images[o][tr] for o in pair}
        Xte_img = {o: images[o][te] for o in pair}
        p = train_multi(TwoBranchConcatNet(sizes, pair, n_classes), Xtr_img, y[tr], Xte_img, binary=False)
        pred = p.argmax(axis=1)
        store["CNN_TwoConcat"]["accuracy"].append(accuracy_score(y[te], pred))
        store["CNN_TwoConcat"]["macro_f1"].append(f1_score(y[te], pred, average="macro"))
        p = train_multi(TwoBranchGatedNet(sizes, pair, n_classes), Xtr_img, y[tr], Xte_img, binary=False)
        pred = p.argmax(axis=1)
        store["CNN_TwoGated"]["accuracy"].append(accuracy_score(y[te], pred))
        store["CNN_TwoGated"]["macro_f1"].append(f1_score(y[te], pred, average="macro"))
        probs = [train_single_prob(Xtr_img[o], y[tr], Xte_img[o], sizes[o], binary=False) for o in pair]
        late = sum(probs) / len(probs)
        pred = late.argmax(axis=1)
        store["CNN_LateAvg"]["accuracy"].append(accuracy_score(y[te], pred))
        store["CNN_LateAvg"]["macro_f1"].append(f1_score(y[te], pred, average="macro"))

    rows = []
    for name, m in store.items():
        rows.append({"pair": "+".join(pair), "scheme": name, "task": "PAM50", "metric": "accuracy", "mean": round(float(np.mean(m["accuracy"])), 4), "std": round(float(np.std(m["accuracy"])), 4)})
        rows.append({"pair": "+".join(pair), "scheme": name, "task": "PAM50", "metric": "macro_f1", "mean": round(float(np.mean(m["macro_f1"])), 4), "std": round(float(np.std(m["macro_f1"])), 4)})
    return rows


def eval_survival(pair, X, images, y_event, y_time):
    sizes = {o: SIZE_SUR[o] for o in pair}
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    store = {}
    for name in ml_models():
        store[f"Concat_{name}"] = {"roc_auc": [], "c_index": []}
    store["Concat_MLP"] = {"roc_auc": [], "c_index": []}
    store["CNN_TwoConcat"] = {"roc_auc": [], "c_index": []}
    store["CNN_TwoGated"] = {"roc_auc": [], "c_index": []}
    store["CNN_LateAvg"] = {"roc_auc": [], "c_index": []}
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        Xtr = np.hstack([X[o][tr] for o in pair])
        Xte = np.hstack([X[o][te] for o in pair])
        for name, clf in ml_models().items():
            pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
            pipe.fit(Xtr, y_event[tr])
            prob = risk_score(pipe[-1], pipe[0].transform(Xte))
            store[f"Concat_{name}"]["roc_auc"].append(roc_auc_score(y_event[te], prob))
            store[f"Concat_{name}"]["c_index"].append(concordance_index(y_time[te], -prob, y_event[te]))
        prob = train_mlp(Xtr, y_event[tr], Xte, binary=True)
        store["Concat_MLP"]["roc_auc"].append(roc_auc_score(y_event[te], prob))
        store["Concat_MLP"]["c_index"].append(concordance_index(y_time[te], -prob, y_event[te]))

        Xtr_img = {o: images[o][tr] for o in pair}
        Xte_img = {o: images[o][te] for o in pair}
        p = train_multi(TwoBranchConcatNet(sizes, pair, 1), Xtr_img, y_event[tr], Xte_img, binary=True)
        store["CNN_TwoConcat"]["roc_auc"].append(roc_auc_score(y_event[te], p))
        store["CNN_TwoConcat"]["c_index"].append(concordance_index(y_time[te], -p, y_event[te]))
        p = train_multi(TwoBranchGatedNet(sizes, pair, 1), Xtr_img, y_event[tr], Xte_img, binary=True)
        store["CNN_TwoGated"]["roc_auc"].append(roc_auc_score(y_event[te], p))
        store["CNN_TwoGated"]["c_index"].append(concordance_index(y_time[te], -p, y_event[te]))
        probs = [train_single_prob(Xtr_img[o], y_event[tr], Xte_img[o], sizes[o], binary=True) for o in pair]
        late = sum(probs) / len(probs)
        store["CNN_LateAvg"]["roc_auc"].append(roc_auc_score(y_event[te], late))
        store["CNN_LateAvg"]["c_index"].append(concordance_index(y_time[te], -late, y_event[te]))

    rows = []
    for name, m in store.items():
        rows.append({"pair": "+".join(pair), "scheme": name, "task": "Survival", "metric": "roc_auc", "mean": round(float(np.mean(m["roc_auc"])), 4), "std": round(float(np.std(m["roc_auc"])), 4)})
        rows.append({"pair": "+".join(pair), "scheme": name, "task": "Survival", "metric": "c_index", "mean": round(float(np.mean(m["c_index"])), 4), "std": round(float(np.std(m["c_index"])), 4)})
    return rows


def main():
    rows = []
    for pair in PAIRS:
        X, images, y, n_classes = load_pam50(pair)
        rows += eval_pam50(pair, X, images, y, n_classes)
        print(f"done PAM50 {pair}", flush=True)
    for pair in PAIRS:
        X, images, y_event, y_time = load_survival(pair)
        rows += eval_survival(pair, X, images, y_event, y_time)
        print(f"done Survival {pair}", flush=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "scheme", "task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
