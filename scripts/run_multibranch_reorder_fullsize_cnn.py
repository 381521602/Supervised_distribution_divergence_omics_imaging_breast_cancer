#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不同 PAM50/Survival mRNA 图谱作为独立分支输入 FullSizeCNN。"""

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
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parent.parent
PAM_DIR = ROOT / "data/final_datasets/PAM50"
SUR_DIR = ROOT / "data/final_datasets/Survival"
IMG = ROOT / "data/images"
OUT = ROOT / "data/multibranch_reorder_fullsize_cnn_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64


class MultiBranchFullSizeCNN(nn.Module):
    def __init__(self, size, out_dim, n_branches):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv2d(1, 32, kernel_size=(size, size)) for _ in range(n_branches)]
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * n_branches, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, xs):
        feats = [F.relu(conv(x)) for conv, x in zip(self.branches, xs)]
        return self.head(torch.cat(feats, dim=1))


def train_pam50(Xtr_list, ytr, Xte_list):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = MultiBranchFullSizeCNN(20, 4, len(Xtr_list))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = [torch.tensor(x, dtype=torch.float32) for x in Xtr_list]
    yt = torch.tensor(ytr, dtype=torch.long)
    n = Xt[0].shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad()
            loss = crit(model([x[idx] for x in Xt]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model([torch.tensor(x, dtype=torch.float32) for x in Xte_list])
    return logits.argmax(dim=1).numpy()


def train_survival(Xtr_list, ytr, Xte_list):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = MultiBranchFullSizeCNN(15, 1, len(Xtr_list))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    Xt = [torch.tensor(x, dtype=torch.float32) for x in Xtr_list]
    yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    n = Xt[0].shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad()
            loss = crit(model([x[idx] for x in Xt]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model([torch.tensor(x, dtype=torch.float32) for x in Xte_list])
    return torch.sigmoid(logits).numpy().ravel()


def image_variants(task):
    base = IMG / task / "mRNA"
    reorder = IMG / task / "mRNA_reorder"
    original = np.load(base / "images.npy")
    s1 = np.load(reorder / "scheme1_function_block/images.npy")
    s2 = np.load(reorder / "scheme2_function_center/images.npy")
    s3 = np.load(reorder / "scheme3_expression_cluster/images.npy")
    return {
        "single_original_nsre": [original],
        "branches_nsre_scheme1": [original, s1],
        "branches_nsre_scheme2": [original, s2],
        "branches_nsre_scheme3": [original, s3],
        "branches_4schemes": [original, s1, s2, s3],
    }


def run_pam50():
    df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
    enc = LabelEncoder()
    y = enc.fit_transform(df["pam50"].values)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    images = image_variants("PAM50")
    results = {name: {"acc": [], "f1": []} for name in images}
    for tr, te in cv.split(np.zeros(len(y)), y):
        for name, xs in images.items():
            pred = train_pam50([x[tr] for x in xs], y[tr], [x[te] for x in xs])
            results[name]["acc"].append(accuracy_score(y[te], pred))
            results[name]["f1"].append(f1_score(y[te], pred, average="macro"))
    rows = []
    for name, m in results.items():
        rows.append({"variant": name, "task": "PAM50", "metric": "accuracy", "mean": round(float(np.mean(m["acc"])), 4), "std": round(float(np.std(m["acc"])), 4)})
        rows.append({"variant": name, "task": "PAM50", "metric": "macro_f1", "mean": round(float(np.mean(m["f1"])), 4), "std": round(float(np.std(m["f1"])), 4)})
    return rows


def run_survival():
    df = pd.read_csv(SUR_DIR / "mRNA_Survival_final.tsv", sep="\t")
    y_event = df["os_event"].astype(int).values
    y_time = df["os_time_days"].astype(float).values
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    images = image_variants("Survival")
    results = {name: {"auc": [], "ci": []} for name in images}
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        for name, xs in images.items():
            prob = train_survival([x[tr] for x in xs], y_event[tr], [x[te] for x in xs])
            results[name]["auc"].append(roc_auc_score(y_event[te], prob))
            results[name]["ci"].append(concordance_index(y_time[te], -prob, y_event[te]))
    rows = []
    for name, m in results.items():
        rows.append({"variant": name, "task": "Survival", "metric": "roc_auc", "mean": round(float(np.mean(m["auc"])), 4), "std": round(float(np.std(m["auc"])), 4)})
        rows.append({"variant": name, "task": "Survival", "metric": "c_index", "mean": round(float(np.mean(m["ci"])), 4), "std": round(float(np.std(m["ci"])), 4)})
    return rows


def main():
    rows = run_pam50() + run_survival()
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}")
    for r in rows:
        print(r["variant"], r["task"], r["metric"], r["mean"], "+-", r["std"])


if __name__ == "__main__":
    main()
