#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NSRE 原始图像上对比几类经典全尺寸卷积结构。"""

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
OUT = ROOT / "data/nsre_classic_structures_results.tsv"
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
        return self.head(F.relu(self.conv(x)))


class GroupNormMishFullSizeCNN(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.gn = nn.GroupNorm(min(8, 32), 32)
        self.head = nn.Sequential(
            nn.Linear(32, 64),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        x = self.gn(self.conv(x))
        x = F.mish(x)
        return self.head(x.flatten(1))


class FCNFullSize(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.bn = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x.flatten(1)


class ResNetFullSize(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.res = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        x = F.relu(self.conv(x)).flatten(1)
        identity = x
        x = F.relu(self.res(x))
        x = x + identity
        return self.head(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = x.mean(dim=(2, 3), keepdim=True).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MultiBranchSEFullSize(nn.Module):
    def __init__(self, size, out_dim, channels=(16, 32, 64)):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv2d(1, c, kernel_size=(size, size)) for c in channels]
        )
        self.se = SEBlock(sum(channels))
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(sum(channels), out_dim),
        )

    def forward(self, x):
        feats = [F.relu(conv(x)) for conv in self.branches]
        z = torch.cat(feats, dim=1)
        z = self.se(z)
        return self.head(z.flatten(1))


def build_model(name, size, out_dim):
    if name == "FullSizeCNN":
        return FullSizeCNN(size, out_dim)
    if name == "GroupNorm_Mish_FullSizeCNN":
        return GroupNormMishFullSizeCNN(size, out_dim)
    if name == "FCN_FullSize":
        return FCNFullSize(size, out_dim)
    if name == "ResNet_FullSize":
        return ResNetFullSize(size, out_dim)
    if name == "MultiBranch_SE_FullSize":
        return MultiBranchSEFullSize(size, out_dim)
    raise ValueError(name)


def train_model(name, size, Xtr, ytr, Xte, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = build_model(name, size, 1 if binary else 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    if binary:
        crit = nn.BCEWithLogitsLoss()
        yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    else:
        crit = nn.CrossEntropyLoss()
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
        out = model(torch.tensor(Xte, dtype=torch.float32))
    return torch.sigmoid(out).numpy().ravel() if binary else out.argmax(dim=1).numpy()


NAMES = [
    "FullSizeCNN",
    "GroupNorm_Mish_FullSizeCNN",
    "FCN_FullSize",
    "ResNet_FullSize",
    "MultiBranch_SE_FullSize",
]


def run_pam50():
    df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
    enc = LabelEncoder()
    y = enc.fit_transform(df["pam50"].values)
    imgs = np.load(IMG / "PAM50/mRNA/images.npy")
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {name: {"acc": [], "f1": []} for name in NAMES}
    for tr, te in cv.split(np.zeros(len(y)), y):
        for name in NAMES:
            pred = train_model(name, 20, imgs[tr], y[tr], imgs[te], binary=False)
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
    imgs = np.load(IMG / "Survival/mRNA/images.npy")
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {name: {"auc": [], "ci": []} for name in NAMES}
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        for name in NAMES:
            prob = train_model(name, 15, imgs[tr], y_event[tr], imgs[te], binary=True)
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
