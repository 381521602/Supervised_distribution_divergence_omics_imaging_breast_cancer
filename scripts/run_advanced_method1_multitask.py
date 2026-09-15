#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Method 1: multi-task PAM50 + Survival joint training."""

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
SUR_DIR = DATA / "final_datasets/Survival"
OUT = DATA / "advanced_method1_multitask_results.tsv"
OMICS = ["mRNA", "CNV", "miRNA"]
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64


class MultiTaskNet(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        self.pam_head = nn.Linear(64, 4)
        self.survival_head = nn.Linear(64, 1)

    def forward(self, x):
        z = self.shared(x)
        return self.pam_head(z), self.survival_head(z)


def load_joint():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    pam_lists = [pd.read_csv(SEL / f"{o}_PAM50_4class_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in OMICS]
    sur_lists = [pd.read_csv(SEL / f"{o}_OS_matrix.tsv", sep="\t", usecols=[0], index_col=0).index.tolist() for o in OMICS]
    pam_common = [c for c in pam_lists[0] if c in set(pam_lists[1]) and c in set(pam_lists[2])]
    sur_common = [c for c in sur_lists[0] if c in set(sur_lists[1]) and c in set(sur_lists[2])]
    common = [c for c in sur_common if c in set(pam_common)]
    X, case_lists = {}, {}
    for o, cases in zip(OMICS, sur_lists):
        df = pd.read_csv(SUR_DIR / f"{o}_Survival_final.tsv", sep="\t")
        Xmat = df.drop(columns=["case_id", "os_event", "os_time_days"]).to_numpy(dtype=float)
        pos = {c: i for i, c in enumerate(cases)}
        idx = np.array([pos[c] for c in common], dtype=int)
        X[o] = Xmat[idx]
    X_all = np.hstack([X[o] for o in OMICS])
    lab = labels.set_index("case_id").loc[common]
    enc = LabelEncoder()
    y_pam = enc.fit_transform(lab["pam50_4class"].values)
    y_event = lab["os_event"].astype(int).values
    y_time = lab["os_time_days"].astype(float).values
    return X_all, y_pam, y_event, y_time, len(enc.classes_)


def main():
    X, y_pam, y_event, y_time, n_classes = load_joint()
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s, aucs, cis = [], [], [], []
    for tr, te in cv.split(np.zeros(len(y_pam)), y_pam):
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        model = MultiTaskNet(X.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.tensor(Xtr, dtype=torch.float32)
        yp = torch.tensor(y_pam[tr], dtype=torch.long)
        ye = torch.tensor(y_event[tr], dtype=torch.float32).view(-1, 1)
        n = Xt.shape[0]
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                if idx.shape[0] < 2:
                    continue
                opt.zero_grad()
                pam_out, sur_out = model(Xt[idx])
                loss = F.cross_entropy(pam_out, yp[idx]) + F.binary_cross_entropy_with_logits(sur_out, ye[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pam_out, sur_out = model(torch.tensor(Xte, dtype=torch.float32))
            pam_pred = pam_out.argmax(dim=1).numpy()
            sur_prob = torch.sigmoid(sur_out).numpy().ravel()
        accs.append(accuracy_score(y_pam[te], pam_pred))
        f1s.append(f1_score(y_pam[te], pam_pred, average="macro"))
        aucs.append(roc_auc_score(y_event[te], sur_prob))
        cis.append(concordance_index(y_time[te], -sur_prob, y_event[te]))
    rows = [
        {"task": "PAM50", "metric": "accuracy", "mean": round(float(np.mean(accs)), 4), "std": round(float(np.std(accs)), 4)},
        {"task": "PAM50", "metric": "macro_f1", "mean": round(float(np.mean(f1s)), 4), "std": round(float(np.std(f1s)), 4)},
        {"task": "Survival", "metric": "roc_auc", "mean": round(float(np.mean(aucs)), 4), "std": round(float(np.std(aucs)), 4)},
        {"task": "Survival", "metric": "c_index", "mean": round(float(np.mean(cis)), 4), "std": round(float(np.std(cis)), 4)},
    ]
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
