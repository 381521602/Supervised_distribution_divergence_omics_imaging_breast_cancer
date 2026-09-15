#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分任务选择：
- PAM50: GroupNorm + Mish FullSizeCNN(32ch) 提取特征 + LogisticRegression
- Survival: GroupNorm + Mish FullSizeCNN(64ch)，保留 MLP 风险头，并额外计算 CNN 特征 + Cox 头
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
PAM_DIR = ROOT / "data/final_datasets/PAM50"
SUR_DIR = ROOT / "data/final_datasets/Survival"
IMG = ROOT / "data/images"
OUT = ROOT / "data/fullsize_task_selection_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 25
BATCH_SIZE = 64


class TaskHeadNet(nn.Module):
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

    def features(self, x):
        z = self.conv(x)
        z = self.gn(z)
        z = F.mish(z)
        return F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)

    def forward(self, x):
        return self.head(self.features(x))


def train_cnn(Xtr, ytr, size, out_dim, channels, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = TaskHeadNet(size, out_dim, channels)
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


def run_pam50():
    imgs = np.load(IMG / "PAM50/mRNA/images.npy")
    df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
    enc = LabelEncoder()
    y = enc.fit_transform(df["pam50"].values)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        cnn = train_cnn(imgs[tr], y[tr], 20, len(enc.classes_), 32, binary=False)
        ftr, fte = extract(cnn, imgs[tr]), extract(cnn, imgs[te])
        scaler = StandardScaler().fit(ftr)
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", class_weight="balanced", random_state=RANDOM_STATE)
        clf.fit(scaler.transform(ftr), y[tr])
        pred = clf.predict(scaler.transform(fte))
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return [
        {"variant": "task_specific", "task": "PAM50", "metric": "accuracy", "mean": round(float(np.mean(accs)), 4), "std": round(float(np.std(accs)), 4)},
        {"variant": "task_specific", "task": "PAM50", "metric": "macro_f1", "mean": round(float(np.mean(f1s)), 4), "std": round(float(np.std(f1s)), 4)},
    ]


def run_survival():
    imgs = np.load(IMG / "Survival/mRNA/images.npy")
    df = pd.read_csv(SUR_DIR / "mRNA_Survival_final.tsv", sep="\t")
    y = df["os_event"].astype(int).values
    y_time = df["os_time_days"].astype(float).values
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mlp_aucs, mlp_cis, cox_aucs, cox_cis = [], [], [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        cnn = train_cnn(imgs[tr], y[tr], 15, 1, 64, binary=True)
        cnn.eval()
        with torch.no_grad():
            prob = torch.sigmoid(cnn(torch.tensor(imgs[te]))).numpy().ravel()
        mlp_aucs.append(roc_auc_score(y[te], prob))
        mlp_cis.append(concordance_index(y_time[te], -prob, y[te]))

        ftr, fte = extract(cnn, imgs[tr]), extract(cnn, imgs[te])
        scaler = StandardScaler().fit(ftr)
        train_df = pd.DataFrame(scaler.transform(ftr), columns=[f"f{i}" for i in range(ftr.shape[1])])
        test_df = pd.DataFrame(scaler.transform(fte), columns=[f"f{i}" for i in range(fte.shape[1])])
        train_df["time"] = y_time[tr].copy()
        train_df["event"] = y[tr].copy()
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(train_df, duration_col="time", event_col="event")
            hazard = cph.predict_partial_hazard(test_df)
            cox_aucs.append(roc_auc_score(y[te], hazard))
            cox_cis.append(concordance_index(y_time[te], -hazard, y[te]))
        except Exception:
            cox_aucs.append(np.nan)
            cox_cis.append(np.nan)
    return [
        {"variant": "task_specific_mlp_head", "task": "Survival", "metric": "roc_auc", "mean": round(float(np.nanmean(mlp_aucs)), 4), "std": round(float(np.nanstd(mlp_aucs)), 4)},
        {"variant": "task_specific_mlp_head", "task": "Survival", "metric": "c_index", "mean": round(float(np.nanmean(mlp_cis)), 4), "std": round(float(np.nanstd(mlp_cis)), 4)},
        {"variant": "task_specific_cox_head", "task": "Survival", "metric": "roc_auc", "mean": round(float(np.nanmean(cox_aucs)), 4), "std": round(float(np.nanstd(cox_aucs)), 4)},
        {"variant": "task_specific_cox_head", "task": "Survival", "metric": "c_index", "mean": round(float(np.nanmean(cox_cis)), 4), "std": round(float(np.nanstd(cox_cis)), 4)},
    ]


def main():
    rows = run_pam50() + run_survival()
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
