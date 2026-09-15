#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四套 PAM50/Survival mRNA 图谱作为多通道输入训练 FullSizeCNN。

multichannel_4schemes: 原始 NSRE + scheme1 + scheme2 + scheme3 堆叠为 4 通道
single_original_nsre: 原始 NSRE 单通道对照
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
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parent.parent
PAM_DIR = ROOT / "data/final_datasets/PAM50"
SUR_DIR = ROOT / "data/final_datasets/Survival"
IMG = ROOT / "data/images"
OUT = ROOT / "data/multichannel_reorder_fullsize_cnn_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64


class FullSizeCNN(nn.Module):
    def __init__(self, size, out_dim, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=(size, size))
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


def train_pam50(Xtr, ytr, Xte, in_channels):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeCNN(20, 4, in_channels)
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


def train_survival(Xtr, ytr, Xte, in_channels):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeCNN(15, 1, in_channels)
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


def load_pam50_images():
    base = IMG / "PAM50/mRNA"
    reorder = IMG / "PAM50/mRNA_reorder"
    return {
        "single_original_nsre": np.load(base / "images.npy"),
        "multichannel_4schemes": np.concatenate(
            [
                np.load(base / "images.npy"),
                np.load(reorder / "scheme1_function_block/images.npy"),
                np.load(reorder / "scheme2_function_center/images.npy"),
                np.load(reorder / "scheme3_expression_cluster/images.npy"),
            ],
            axis=1,
        ),
    }


def load_survival_images():
    base = IMG / "Survival/mRNA"
    reorder = IMG / "Survival/mRNA_reorder"
    return {
        "single_original_nsre": np.load(base / "images.npy"),
        "multichannel_4schemes": np.concatenate(
            [
                np.load(base / "images.npy"),
                np.load(reorder / "scheme1_function_block/images.npy"),
                np.load(reorder / "scheme2_function_center/images.npy"),
                np.load(reorder / "scheme3_expression_cluster/images.npy"),
            ],
            axis=1,
        ),
    }


def run_pam50():
    df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
    enc = LabelEncoder()
    y = enc.fit_transform(df["pam50"].values)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    images = load_pam50_images()
    results = {name: {"acc": [], "f1": []} for name in images}
    for tr, te in cv.split(np.zeros(len(y)), y):
        for name, imgs in images.items():
            pred = train_pam50(imgs[tr], y[tr], imgs[te], imgs.shape[1])
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
    images = load_survival_images()
    results = {name: {"auc": [], "ci": []} for name in images}
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        for name, imgs in images.items():
            prob = train_survival(imgs[tr], y_event[tr], imgs[te], imgs.shape[1])
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
