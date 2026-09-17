#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dense-equivalent baseline for FullSizeCNN.

FullSizeCNN uses a single full-size convolutional layer (1 -> 32 filters) that
produces 32 spatial outputs, then Flatten -> Dense(32 -> 64) -> ReLU -> Dropout
-> Dense(64 -> out). The dense-equivalent model removes the spatial constraint
but keeps the parameter count of the first layer identical:

    Flatten(x) -> Dense(n_features -> 32) -> ReLU
               -> Dense(32 -> 64) -> ReLU -> Dropout(0.3) -> Dense(64 -> out)

The first layer has n_features*32+32 parameters, exactly equal to the full-size
conv layer (1*32*size*size+32 with size*size = n_features). This isolates the
contribution of the learned 2-D spatial arrangement relative to a plain dense
network of the same capacity.
"""

from __future__ import annotations

import csv
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
PAM_DIR = ROOT / "data/final_datasets/PAM50"
SUR_DIR = ROOT / "data/final_datasets/Survival"
OUT = ROOT / "data/dense_equivalent_baseline_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
OMICS = ["mRNA", "CNV", "miRNA"]


class DenseEquivalent(nn.Module):
    def __init__(self, din, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_model(Xtr, ytr, Xte, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = DenseEquivalent(Xtr.shape[1], 1 if binary else 4)
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


def eval_pam50(omics, X, y):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    acc, f1 = [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        p = train_model(Xtr, y[tr], Xte, binary=False)
        pred = p.argmax(axis=1)
        acc.append(accuracy_score(y[te], pred))
        f1.append(f1_score(y[te], pred, average="macro"))
    return [
        {"omics": omics, "task": "PAM50", "metric": "accuracy", "mean": round(float(np.mean(acc)), 4), "std": round(float(np.std(acc)), 4)},
        {"omics": omics, "task": "PAM50", "metric": "macro_f1", "mean": round(float(np.mean(f1)), 4), "std": round(float(np.std(f1)), 4)},
    ]


def eval_survival(omics, X, y_event, y_time):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    auc, ci = [], []
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        prob = train_model(Xtr, y_event[tr], Xte, binary=True)
        auc.append(roc_auc_score(y_event[te], prob))
        ci.append(concordance_index(y_time[te], -prob, y_event[te]))
    return [
        {"omics": omics, "task": "Survival", "metric": "roc_auc", "mean": round(float(np.mean(auc)), 4), "std": round(float(np.std(auc)), 4)},
        {"omics": omics, "task": "Survival", "metric": "c_index", "mean": round(float(np.mean(ci)), 4), "std": round(float(np.std(ci)), 4)},
    ]


def main():
    rows = []
    for omics in OMICS:
        df = pd.read_csv(PAM_DIR / f"{omics}_PAM50_final.tsv", sep="\t")
        X = df.drop(columns=["pam50"]).to_numpy(dtype=float)
        enc = LabelEncoder()
        y = enc.fit_transform(df["pam50"].values)
        rows += eval_pam50(omics, X, y)
        print(f"done PAM50 {omics}", flush=True)

    for omics in OMICS:
        df = pd.read_csv(SUR_DIR / f"{omics}_Survival_final.tsv", sep="\t")
        X = df.drop(columns=["case_id", "os_event", "os_time_days"]).to_numpy(dtype=float)
        y_event = df["os_event"].astype(int).values
        y_time = df["os_time_days"].astype(float).values
        rows += eval_survival(omics, X, y_event, y_time)
        print(f"done Survival {omics}", flush=True)

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["omics", "task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
