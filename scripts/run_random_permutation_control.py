#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random-permutation ordering control for FullSizeCNN.

Compares the deterministic JSD/JSD spiral ordering against:
  - mean-expression ordering (descending),
  - N random permutations of the gene order (null distribution).

Only mRNA is used because the imaging/ordering claim centers on mRNA. Both PAM50
and Survival tasks are evaluated with the same FullSizeCNN, 5-fold CV, seed 42.
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
OUT = ROOT / "data/random_permutation_control_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
N_RANDOM = 20


class FullSizeCNN(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.head(F.relu(self.conv(x)))


def spiral_order(size):
    positions = []
    r = c = size // 2
    positions.append((r, c))
    step = 1
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    d = 0
    while len(positions) < size * size:
        for _ in range(2):
            dr, dc = directions[d % 4]
            for _ in range(step):
                r += dr
                c += dc
                if 0 <= r < size and 0 <= c < size:
                    positions.append((r, c))
            d += 1
        step += 1
    return positions[: size * size]


def make_images_from_order(X, order, size):
    n = X.shape[0]
    positions = spiral_order(size)
    imgs = np.zeros((n, 1, size, size), dtype=np.float32)
    for i in range(n):
        for (r, c), j in zip(positions, order):
            if j < X.shape[1]:
                imgs[i, 0, r, c] = X[i, j]
    return imgs


def train_cnn(imgs_tr, ytr, imgs_te, size, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeCNN(size, 1 if binary else 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(imgs_tr, dtype=torch.float32)
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
        out = model(torch.tensor(imgs_te, dtype=torch.float32))
    if binary:
        return torch.sigmoid(out).numpy().ravel()
    return F.softmax(out, dim=1).numpy()


def evaluate(imgs, y, size, binary, y_time=None):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    acc, f1, auc, ci = [], [], [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        if binary:
            prob = train_cnn(imgs[tr], y[tr], imgs[te], size, True)
            auc.append(roc_auc_score(y[te], prob))
            ci.append(concordance_index(y_time[te], -prob, y[te]))
        else:
            p = train_cnn(imgs[tr], y[tr], imgs[te], size, False)
            pred = p.argmax(axis=1)
            acc.append(accuracy_score(y[te], pred))
            f1.append(f1_score(y[te], pred, average="macro"))
    if binary:
        return {"roc_auc": (float(np.mean(auc)), float(np.std(auc))), "c_index": (float(np.mean(ci)), float(np.std(ci)))}
    return {"accuracy": (float(np.mean(acc)), float(np.std(acc))), "macro_f1": (float(np.mean(f1)), float(np.std(f1)))}


def run_task(X, y, size, binary, y_time=None, task="PAM50"):
    rows = []
    # mean-expression ordering (descending mean across samples)
    expr_order = np.argsort(np.nanmean(X, axis=0))[::-1]
    imgs = make_images_from_order(X, expr_order, size)
    r = evaluate(imgs, y, size, binary, y_time)
    for metric, (m, s) in r.items():
        rows.append({"task": task, "ordering": "mean_expression", "repeat": -1, "metric": metric, "mean": round(m, 4), "std": round(s, 4)})
    print(f"{task} mean_expression done", flush=True)

    # random permutations (null)
    rng = np.random.RandomState(RANDOM_STATE)
    for rep in range(N_RANDOM):
        order = rng.permutation(X.shape[1])
        imgs = make_images_from_order(X, order, size)
        r = evaluate(imgs, y, size, binary, y_time)
        for metric, (m, s) in r.items():
            rows.append({"task": task, "ordering": "random", "repeat": rep, "metric": metric, "mean": round(m, 4), "std": round(s, 4)})
        print(f"{task} random {rep + 1}/{N_RANDOM} done", flush=True)
    return rows


def main():
    rows = []
    # PAM50 mRNA
    df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
    X = df.drop(columns=["pam50"]).to_numpy(dtype=float)
    enc = LabelEncoder()
    y = enc.fit_transform(df["pam50"].values)
    rows += run_task(X, y, 20, binary=False, task="PAM50")

    # Survival mRNA
    df = pd.read_csv(SUR_DIR / "mRNA_Survival_final.tsv", sep="\t")
    X = df.drop(columns=["case_id", "os_event", "os_time_days"]).to_numpy(dtype=float)
    y = df["os_event"].astype(int).values
    y_time = df["os_time_days"].astype(float).values
    rows += run_task(X, y, 15, binary=True, y_time=y_time, task="Survival")

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "ordering", "repeat", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
