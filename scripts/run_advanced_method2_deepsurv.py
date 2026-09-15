#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Method 2: DeepSurv-style Cox deep integration."""

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
SEL = DATA / "selected_features"
SUR_DIR = DATA / "final_datasets/Survival"
OUT = DATA / "advanced_method2_deepsurv_results.tsv"
OMICS = ["mRNA", "CNV", "miRNA"]
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 64


class DeepSurvNet(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


def cox_loss(risk, time, event):
    order = torch.argsort(-time)
    risk = risk[order]
    event = event[order]
    time = time[order]
    loss = risk[0] * 0.0
    n = 0
    for i in range(len(time)):
        if event[i] == 0:
            continue
        risk_i = risk[i]
        risk_set = risk[i:]
        loss = loss - risk_i + torch.logsumexp(risk_set, dim=0)
        n += 1
    return loss / max(n, 1)


def load_survival():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    lists = [pd.read_csv(SEL / f"{o}_OS_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in OMICS]
    common = [c for c in lists[0] if c in set(lists[1]) and c in set(lists[2])]
    X = []
    for o, cases in zip(OMICS, lists):
        df = pd.read_csv(SUR_DIR / f"{o}_Survival_final.tsv", sep="\t")
        Xmat = df.drop(columns=["case_id", "os_event", "os_time_days"]).to_numpy(dtype=float)
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X.append(Xmat[idx])
    X = np.hstack(X)
    lab = labels.set_index("case_id").loc[common]
    return X, lab["os_event"].astype(int).values, lab["os_time_days"].astype(float).values


def main():
    X, y_event, y_time = load_survival()
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, cis = [], []
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        model = DeepSurvNet(X.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.tensor(Xtr, dtype=torch.float32)
        time_t = torch.tensor(y_time[tr], dtype=torch.float32)
        event_t = torch.tensor(y_event[tr], dtype=torch.float32)
        n = Xt.shape[0]
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                if idx.shape[0] < 2:
                    continue
                opt.zero_grad()
                risk = model(Xt[idx]).squeeze(1)
                loss = cox_loss(risk, time_t[idx], event_t[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            hazard = model(torch.tensor(Xte, dtype=torch.float32)).squeeze(1).numpy()
        aucs.append(roc_auc_score(y_event[te], hazard))
        cis.append(concordance_index(y_time[te], -hazard, y_event[te]))
    rows = [
        {"metric": "roc_auc", "mean": round(float(np.mean(aucs)), 4), "std": round(float(np.std(aucs)), 4)},
        {"metric": "c_index", "mean": round(float(np.mean(cis)), 4), "std": round(float(np.std(cis)), 4)},
    ]
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
