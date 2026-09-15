#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Method 5: simplified GNN + co-expression prior."""

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
OUT = DATA / "advanced_method5_gnn_results.tsv"
OMICS = ["mRNA", "CNV", "miRNA"]
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64


class GCNFusion(nn.Module):
    def __init__(self, d_mrna, d_cnv, d_mirna, out_dim):
        super().__init__()
        self.gcn1 = nn.Linear(1, 32)
        self.gcn2 = nn.Linear(32, 32)
        self.proj_cnv = nn.Linear(d_cnv, 32)
        self.proj_mirna = nn.Linear(d_mirna, 32)
        self.head = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, x_mrna, x_cnv, x_mirna, A):
        h = self.gcn1(x_mrna.unsqueeze(-1))
        h = F.relu(torch.matmul(A, h))
        h = self.gcn2(h)
        h = F.relu(torch.matmul(A, h))
        h_mrna = h.mean(dim=1)
        h_cnv = self.proj_cnv(x_cnv)
        h_mirna = self.proj_mirna(x_mirna)
        return self.head(torch.cat([h_mrna, h_cnv, h_mirna], dim=1))


def build_adjacency(Xtr, topk=40):
    corr = np.corrcoef(Xtr.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.abs(corr)
    n = A.shape[0]
    for i in range(n):
        idx = np.argsort(A[i])[::-1][:topk]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        A[i, ~mask] = 0
    A = A + np.eye(n)
    d = A.sum(axis=1, keepdims=True)
    d = np.where(d == 0, 1, d)
    A = A / np.sqrt(d * d.T)
    return torch.tensor(A, dtype=torch.float32)


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


def train(Xtr, ytr, Xte, A, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = GCNFusion(Xtr["mRNA"].shape[1], Xtr["CNV"].shape[1], Xtr["miRNA"].shape[1], 1 if binary else 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtm = torch.tensor(Xtr["mRNA"], dtype=torch.float32)
    Xtc = torch.tensor(Xtr["CNV"], dtype=torch.float32)
    Xtmi = torch.tensor(Xtr["miRNA"], dtype=torch.float32)
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
            loss = crit(model(Xtm[idx], Xtc[idx], Xtmi[idx], A), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte["mRNA"], dtype=torch.float32), torch.tensor(Xte["CNV"], dtype=torch.float32), torch.tensor(Xte["miRNA"], dtype=torch.float32), A)
    return torch.sigmoid(out).numpy().ravel() if binary else F.softmax(out, dim=1).numpy()


def run_task(task, X, y, y_time=None):
    binary = task == "Survival"
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s, aucs, cis = [], [], [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        scalers = {o: StandardScaler().fit(X[o][tr]) for o in OMICS}
        Xtr = {o: scalers[o].transform(X[o][tr]) for o in OMICS}
        Xte = {o: scalers[o].transform(X[o][te]) for o in OMICS}
        A = build_adjacency(Xtr["mRNA"])
        p = train(Xtr, y[tr], Xte, A, binary)
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
    rows = run_task("PAM50", X, y)
    X, y_event, y_time = load_survival()
    rows += run_task("Survival", X, y_event, y_time)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
