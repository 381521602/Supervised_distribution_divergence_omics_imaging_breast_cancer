#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比不同 Survival mRNA 图像排序下的基础 FullSizeCNN。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent.parent
SUR_DIR = ROOT / "data/final_datasets/Survival"
IMG = ROOT / "data/images/Survival/mRNA"
REORDER = ROOT / "data/images/Survival/mRNA_reorder"
OUT = ROOT / "data/reorder_fullsize_cnn_survival_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64


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
        x = self.conv(x)
        x = F.relu(x)
        return self.head(x)


def train_fold(Xtr, ytr, Xte):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeCNN(15, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    n = Xt.shape[0]
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
        logits = model(torch.tensor(Xte, dtype=torch.float32))
    return torch.sigmoid(logits).numpy().ravel()


def evaluate(images, y_event, y_time):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, cis = [], []
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        prob = train_fold(images[tr], y_event[tr], images[te])
        aucs.append(roc_auc_score(y_event[te], prob))
        cis.append(concordance_index(y_time[te], -prob, y_event[te]))
    return round(float(np.mean(aucs)), 4), round(float(np.std(aucs)), 4), round(float(np.mean(cis)), 4), round(float(np.std(cis)), 4)


def main():
    df = pd.read_csv(SUR_DIR / "mRNA_Survival_final.tsv", sep="\t")
    y_event = df["os_event"].astype(int).values
    y_time = df["os_time_days"].astype(float).values

    variants = [
        ("original_jsd_spiral", IMG / "images.npy"),
        ("scheme1_function_block", REORDER / "scheme1_function_block/images.npy"),
        ("scheme2_function_center", REORDER / "scheme2_function_center/images.npy"),
        ("scheme3_expression_cluster", REORDER / "scheme3_expression_cluster/images.npy"),
    ]

    rows = []
    for name, path in variants:
        images = np.load(path)
        auc, auc_std, ci, ci_std = evaluate(images, y_event, y_time)
        rows.append({"variant": name, "task": "Survival", "metric": "roc_auc", "mean": auc, "std": auc_std})
        rows.append({"variant": name, "task": "Survival", "metric": "c_index", "mean": ci, "std": ci_std})
        print(name, "auc", auc, "cindex", ci)

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
