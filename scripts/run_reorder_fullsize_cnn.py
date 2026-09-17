#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比不同 PAM50 mRNA 图像排序下的基础 FullSizeCNN。

variants:
- original_jsd_spiral: 原始 JSD 螺旋排序
- scheme1_function_block
- scheme2_function_center
- scheme3_expression_cluster
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parent.parent
PAM_DIR = ROOT / "data/final_datasets/PAM50"
IMG = ROOT / "data/images/PAM50/mRNA"
REORDER = ROOT / "data/images/PAM50/mRNA_reorder"
OUT = ROOT / "data/reorder_fullsize_cnn_results.tsv"
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


def train_fold(Xtr, ytr, Xte, size, out_dim):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeCNN(size, out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
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
    return logits.argmax(dim=1).numpy()


def evaluate(images, y, enc):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        pred = train_fold(images[tr], y[tr], images[te], 20, len(enc.classes_))
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4), round(float(np.mean(f1s)), 4), round(float(np.std(f1s)), 4)


def main():
    df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
    enc = LabelEncoder()
    y = enc.fit_transform(df["pam50"].values)

    variants = [
        ("original_jsd_spiral", IMG / "images.npy"),
        ("scheme1_function_block", REORDER / "scheme1_function_block/images.npy"),
        ("scheme2_function_center", REORDER / "scheme2_function_center/images.npy"),
        ("scheme3_expression_cluster", REORDER / "scheme3_expression_cluster/images.npy"),
    ]

    rows = []
    for name, path in variants:
        images = np.load(path)
        acc, acc_std, f1, f1_std = evaluate(images, y, enc)
        rows.append({"variant": name, "task": "PAM50", "metric": "accuracy", "mean": acc, "std": acc_std})
        rows.append({"variant": name, "task": "PAM50", "metric": "macro_f1", "mean": f1, "std": f1_std})
        print(name, "acc", acc, "f1", f1)

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
