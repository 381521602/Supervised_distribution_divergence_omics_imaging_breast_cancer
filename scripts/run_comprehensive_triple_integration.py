#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive batch 2a: triple-omics integration on the three-omics intersection."""

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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_fullsize_multimodal_integration import Branch, TripleConcatNet, TripleGatedNet, train_multi, train_single_prob  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
SEL = DATA / "selected_features"
PAM_DIR = DATA / "final_datasets/PAM50"
SUR_DIR = DATA / "final_datasets/Survival"
IMG = DATA / "images"
OUT = DATA / "comprehensive_triple_integration_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
OMICS = ["mRNA", "CNV", "miRNA"]


def ml_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVC": SVC(kernel="linear", probability=False, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


class MLP(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, dout),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, Xte, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = MLP(Xtr.shape[1], 1 if binary else 4)
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


def risk_score(clf, Xte):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(Xte)[:, 1]
    return clf.decision_function(Xte)


def load_pam50():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    case_lists = [pd.read_csv(SEL / f"{o}_PAM50_4class_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in OMICS]
    common = [c for c in case_lists[0] if c in set(case_lists[1]) and c in set(case_lists[2])]
    sizes = {"mRNA": 20, "CNV": 8, "miRNA": 25}
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
    return X, images, y, sizes, len(enc.classes_)


def load_survival():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    case_lists = [pd.read_csv(SEL / f"{o}_OS_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in OMICS]
    common = [c for c in case_lists[0] if c in set(case_lists[1]) and c in set(case_lists[2])]
    sizes = {"mRNA": 15, "CNV": 13, "miRNA": 25}
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
    return X, images, y_event, y_time, sizes


def eval_pam50(X, images, y, sizes, n_classes):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    store = {}
    for name in ml_models():
        store[f"Concat_{name}"] = {"accuracy": [], "macro_f1": []}
    store["Concat_MLP"] = {"accuracy": [], "macro_f1": []}
    store["CNN_TripleConcat"] = {"accuracy": [], "macro_f1": []}
    store["CNN_TripleGated"] = {"accuracy": [], "macro_f1": []}
    store["CNN_LateAvg"] = {"accuracy": [], "macro_f1": []}

    for tr, te in cv.split(np.zeros(len(y)), y):
        Xtr = np.hstack([X[o][tr] for o in OMICS])
        Xte = np.hstack([X[o][te] for o in OMICS])
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

        Xtr_img = {o: images[o][tr] for o in OMICS}
        Xte_img = {o: images[o][te] for o in OMICS}
        p = train_multi(TripleConcatNet(sizes, n_classes), Xtr_img, y[tr], Xte_img, binary=False)
        pred = p.argmax(axis=1)
        store["CNN_TripleConcat"]["accuracy"].append(accuracy_score(y[te], pred))
        store["CNN_TripleConcat"]["macro_f1"].append(f1_score(y[te], pred, average="macro"))
        p = train_multi(TripleGatedNet(sizes, n_classes), Xtr_img, y[tr], Xte_img, binary=False)
        pred = p.argmax(axis=1)
        store["CNN_TripleGated"]["accuracy"].append(accuracy_score(y[te], pred))
        store["CNN_TripleGated"]["macro_f1"].append(f1_score(y[te], pred, average="macro"))
        probs = [train_single_prob(Xtr_img[o], y[tr], Xte_img[o], sizes[o], binary=False) for o in OMICS]
        late = sum(probs) / len(probs)
        pred = late.argmax(axis=1)
        store["CNN_LateAvg"]["accuracy"].append(accuracy_score(y[te], pred))
        store["CNN_LateAvg"]["macro_f1"].append(f1_score(y[te], pred, average="macro"))

    rows = []
    for name, m in store.items():
        rows.append({"scheme": name, "task": "PAM50", "metric": "accuracy", "mean": round(float(np.mean(m["accuracy"])), 4), "std": round(float(np.std(m["accuracy"])), 4)})
        rows.append({"scheme": name, "task": "PAM50", "metric": "macro_f1", "mean": round(float(np.mean(m["macro_f1"])), 4), "std": round(float(np.std(m["macro_f1"])), 4)})
    return rows


def eval_survival(X, images, y_event, y_time, sizes):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    store = {}
    for name in ml_models():
        store[f"Concat_{name}"] = {"roc_auc": [], "c_index": []}
    store["Concat_MLP"] = {"roc_auc": [], "c_index": []}
    store["CNN_TripleConcat"] = {"roc_auc": [], "c_index": []}
    store["CNN_TripleGated"] = {"roc_auc": [], "c_index": []}
    store["CNN_LateAvg"] = {"roc_auc": [], "c_index": []}

    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        Xtr = np.hstack([X[o][tr] for o in OMICS])
        Xte = np.hstack([X[o][te] for o in OMICS])
        for name, clf in ml_models().items():
            pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
            pipe.fit(Xtr, y_event[tr])
            prob = risk_score(pipe[-1], pipe[0].transform(Xte))
            store[f"Concat_{name}"]["roc_auc"].append(roc_auc_score(y_event[te], prob))
            store[f"Concat_{name}"]["c_index"].append(concordance_index(y_time[te], -prob, y_event[te]))
        prob = train_mlp(Xtr, y_event[tr], Xte, binary=True)
        store["Concat_MLP"]["roc_auc"].append(roc_auc_score(y_event[te], prob))
        store["Concat_MLP"]["c_index"].append(concordance_index(y_time[te], -prob, y_event[te]))

        Xtr_img = {o: images[o][tr] for o in OMICS}
        Xte_img = {o: images[o][te] for o in OMICS}
        p = train_multi(TripleConcatNet(sizes, 1), Xtr_img, y_event[tr], Xte_img, binary=True)
        store["CNN_TripleConcat"]["roc_auc"].append(roc_auc_score(y_event[te], p))
        store["CNN_TripleConcat"]["c_index"].append(concordance_index(y_time[te], -p, y_event[te]))
        p = train_multi(TripleGatedNet(sizes, 1), Xtr_img, y_event[tr], Xte_img, binary=True)
        store["CNN_TripleGated"]["roc_auc"].append(roc_auc_score(y_event[te], p))
        store["CNN_TripleGated"]["c_index"].append(concordance_index(y_time[te], -p, y_event[te]))
        probs = [train_single_prob(Xtr_img[o], y_event[tr], Xte_img[o], sizes[o], binary=True) for o in OMICS]
        late = sum(probs) / len(probs)
        store["CNN_LateAvg"]["roc_auc"].append(roc_auc_score(y_event[te], late))
        store["CNN_LateAvg"]["c_index"].append(concordance_index(y_time[te], -late, y_event[te]))

    rows = []
    for name, m in store.items():
        rows.append({"scheme": name, "task": "Survival", "metric": "roc_auc", "mean": round(float(np.mean(m["roc_auc"])), 4), "std": round(float(np.std(m["roc_auc"])), 4)})
        rows.append({"scheme": name, "task": "Survival", "metric": "c_index", "mean": round(float(np.mean(m["c_index"])), 4), "std": round(float(np.std(m["c_index"])), 4)})
    return rows


def main():
    X, images, y, sizes, n_classes = load_pam50()
    rows = eval_pam50(X, images, y, sizes, n_classes)
    print("done PAM50", flush=True)
    X, images, y_event, y_time, sizes = load_survival()
    rows += eval_survival(X, images, y_event, y_time, sizes)
    print("done Survival", flush=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scheme", "task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
