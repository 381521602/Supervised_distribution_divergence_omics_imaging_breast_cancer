#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CNN 特征提取 + LogisticRegression（GroupNorm + Mish FullSizeCNN）。

将 FullSizeCNN 作为图像特征提取器，在每一折内：
1. 只在训练折上训练 CNN；
2. 用训练好的 CNN 提取训练/测试折特征；
3. 在训练折特征上拟合 StandardScaler + LogisticRegression；
4. 在测试折上评估。
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
PAM_DIR = ROOT / "data/final_datasets/PAM50"
SUR_DIR = ROOT / "data/final_datasets/Survival"
IMG = ROOT / "data/images"
OUT = ROOT / "data/fullsize_lr_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 25
BATCH_SIZE = 64


class FullSizeFeatureNet(nn.Module):
    """GroupNorm + Mish 全尺寸卷积特征提取器，附带一个小型分类头用于训练。"""

    def __init__(self, size: int, out_dim: int, channels: int):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(1, channels, kernel_size=(size, size))
        self.gn = nn.GroupNorm(min(8, channels), channels)
        self.head = nn.Sequential(
            nn.Linear(channels, 64),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        z = self.gn(z)
        z = F.mish(z)
        z = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def train_cnn(Xtr, ytr, size, out_dim, channels, binary=False):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeFeatureNet(size, out_dim, channels)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if binary:
        crit = nn.BCEWithLogitsLoss()
        yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    else:
        crit = nn.CrossEntropyLoss()
        yt = torch.tensor(ytr, dtype=torch.long)
    Xt = torch.tensor(Xtr)
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
    return model


def extract(model, X):
    model.eval()
    with torch.no_grad():
        return model.features(torch.tensor(X)).numpy()


def eval_cnn_lr(channels, task):
    size = 20 if task == "PAM50" else 15
    imgs = np.load(IMG / f"{task}/mRNA/images.npy")
    if task == "PAM50":
        df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
        y_raw = df["pam50"].values
        enc = LabelEncoder()
        y = enc.fit_transform(y_raw)
    else:
        df = pd.read_csv(SUR_DIR / "mRNA_Survival_final.tsv", sep="\t")
        y = df["os_event"].astype(int).values

    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    variant = f"cnn_lr_ch{channels}"

    if task == "PAM50":
        accs, f1s = [], []
        for tr, te in cv.split(np.zeros(len(y)), y):
            cnn = train_cnn(imgs[tr], y[tr], size, len(enc.classes_), channels)
            ftr, fte = extract(cnn, imgs[tr]), extract(cnn, imgs[te])
            scaler = StandardScaler().fit(ftr)
            clf = LogisticRegression(
                C=1.0,
                max_iter=5000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ).fit(scaler.transform(ftr), y[tr])
            pred = clf.predict(scaler.transform(fte))
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="macro"))
        return [
            {"variant": variant, "task": "PAM50", "metric": "accuracy", "mean": round(float(np.mean(accs)), 4), "std": round(float(np.std(accs)), 4)},
            {"variant": variant, "task": "PAM50", "metric": "macro_f1", "mean": round(float(np.mean(f1s)), 4), "std": round(float(np.std(f1s)), 4)},
        ]

    y_time = df["os_time_days"].astype(float).values
    aucs, cis = [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        cnn = train_cnn(imgs[tr], y[tr], size, 1, channels, binary=True)
        ftr, fte = extract(cnn, imgs[tr]), extract(cnn, imgs[te])
        scaler = StandardScaler().fit(ftr)
        clf = LogisticRegression(
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ).fit(scaler.transform(ftr), y[tr])
        prob = clf.predict_proba(scaler.transform(fte))[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
        cis.append(concordance_index(y_time[te], -prob, y[te]))
    return [
        {"variant": variant, "task": "Survival", "metric": "roc_auc", "mean": round(float(np.mean(aucs)), 4), "std": round(float(np.std(aucs)), 4)},
        {"variant": variant, "task": "Survival", "metric": "c_index", "mean": round(float(np.mean(cis)), 4), "std": round(float(np.std(cis)), 4)},
    ]


def main():
    rows = []
    for channels in (32, 64):
        rows += eval_cnn_lr(channels, "PAM50")
        rows += eval_cnn_lr(channels, "Survival")
        print("done", channels)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
