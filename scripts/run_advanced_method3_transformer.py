#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Method 3: Transformer cross-omics attention."""

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
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
SEL = DATA / "selected_features"
PAM_DIR = DATA / "final_datasets/PAM50"
SUR_DIR = DATA / "final_datasets/Survival"
OUT = DATA / "advanced_method3_transformer_results.tsv"
OMICS = ["mRNA", "CNV", "miRNA"]
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64


class TransformerFusion(nn.Module):
    def __init__(self, dims, d_model=32, nhead=4, layers=2, out_dim=1):
        super().__init__()
        self.proj = nn.ModuleDict({o: nn.Linear(dims[o], d_model) for o in OMICS})
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, xs):
        tokens = torch.stack([self.proj[o](xs[o]) for o in OMICS], dim=1)
        z = self.encoder(tokens)
        z = z.mean(dim=1)
        return self.head(z)


def load_pam50():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    lists = [pd.read_csv(SEL / f"{o}_PAM50_4class_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in OMICS]
    common = [c for c in lists[0] if c in set(lists[1]) and c in set(lists[2])]
    X = {}
    for o, cases in zip(OMICS, lists):
        df = pd.read_csv(PAM_DIR / f"{o}_PAM50_final.tsv", sep="\t")
        Xmat = df.drop(columns=["pam50"]).to_numpy(dtype=float)
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X[o] = Xmat[idx]
    y_raw = labels.set_index("case_id").loc[common, "pam50_4class"].values
    enc = LabelEncoder()
    return X, enc.fit_transform(y_raw), len(enc.classes_)


def load_survival():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    lists = [pd.read_csv(SEL / f"{o}_OS_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in OMICS]
    common = [c for c in lists[0] if c in set(lists[1]) and c in set(lists[2])]
    X = {}
    for o, cases in zip(OMICS, lists):
        df = pd.read_csv(SUR_DIR / f"{o}_Survival_final.tsv", sep="\t")
        Xmat = df.drop(columns=["case_id", "os_event", "os_time_days"]).to_numpy(dtype=float)
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X[o] = Xmat[idx]
    lab = labels.set_index("case_id").loc[common]
    return X, lab["os_event"].astype(int).values, lab["os_time_days"].astype(float).values


def train(Xtr, ytr, Xte, dims, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = TransformerFusion(dims, out_dim=1 if binary else 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = {o: torch.tensor(v, dtype=torch.float32) for o, v in Xtr.items()}
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
            loss = crit(model({o: v[idx] for o, v in Xt.items()}), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model({o: torch.tensor(v, dtype=torch.float32) for o, v in Xte.items()})
    return torch.sigmoid(out).numpy().ravel() if binary else F.softmax(out, dim=1).numpy()


def run_task(task, X, y, dims, y_time=None):
    binary = task == "Survival"
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s, aucs, cis = [], [], [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        scalers = {o: StandardScaler().fit(X[o][tr]) for o in OMICS}
        Xtr = {o: scalers[o].transform(X[o][tr]) for o in OMICS}
        Xte = {o: scalers[o].transform(X[o][te]) for o in OMICS}
        p = train(Xtr, y[tr], Xte, dims, binary)
        if binary:
            aucs.append(roc_auc_score(y[te], p))
            cis.append(concordance_index(y_time[te], -p, y[te]))
        else:
            pred = p.argmax(axis=1)
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="macro"))
    rows = []
    if binary:
        rows.append({"task": task, "metric": "roc_auc", "mean": round(float(np.mean(aucs)), 4), "std": round(float(np.std(aucs)), 4)})
        rows.append({"task": task, "metric": "c_index", "mean": round(float(np.mean(cis)), 4), "std": round(float(np.std(cis)), 4)})
    else:
        rows.append({"task": task, "metric": "accuracy", "mean": round(float(np.mean(accs)), 4), "std": round(float(np.std(accs)), 4)})
        rows.append({"task": task, "metric": "macro_f1", "mean": round(float(np.mean(f1s)), 4), "std": round(float(np.std(f1s)), 4)})
    return rows


def main():
    X, y, n_classes = load_pam50()
    dims = {o: X[o].shape[1] for o in OMICS}
    rows = run_task("PAM50", X, y, dims)
    X, y_event, y_time = load_survival()
    dims = {o: X[o].shape[1] for o in OMICS}
    rows += run_task("Survival", X, y_event, dims, y_time)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
