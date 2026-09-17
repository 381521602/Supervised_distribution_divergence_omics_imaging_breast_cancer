#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对四种 PAM50/Survival mRNA 图像排序方案做配对检验。

所有方案使用同一 5 折切分，逐折保存相同折上的 Accuracy/Macro-F1/ROC AUC/C-index，
再进行两两配对 t 检验与 Wilcoxon signed-rank 检验。
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parent.parent
PAM_DIR = ROOT / "data/final_datasets/PAM50"
SUR_DIR = ROOT / "data/final_datasets/Survival"
IMG = ROOT / "data/images"
OUT = ROOT / "data/reorder_paired_test_results.tsv"
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


def train_pam50(Xtr, ytr, Xte):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeCNN(20, 4)
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


def train_survival(Xtr, ytr, Xte):
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


def variants_for(task: str):
    base = IMG / task / "mRNA"
    reorder = IMG / task / "mRNA_reorder"
    return [
        ("original_jsd_spiral", base / "images.npy"),
        ("scheme1_function_block", reorder / "scheme1_function_block/images.npy"),
        ("scheme2_function_center", reorder / "scheme2_function_center/images.npy"),
        ("scheme3_expression_cluster", reorder / "scheme3_expression_cluster/images.npy"),
    ]


def run_pam50():
    df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
    enc = LabelEncoder()
    y = enc.fit_transform(df["pam50"].values)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    data = {name: {"accuracy": [], "macro_f1": []} for name, _ in variants_for("PAM50")}
    images = {name: np.load(path) for name, path in variants_for("PAM50")}
    for tr, te in cv.split(np.zeros(len(y)), y):
        for name, imgs in images.items():
            pred = train_pam50(imgs[tr], y[tr], imgs[te])
            data[name]["accuracy"].append(accuracy_score(y[te], pred))
            data[name]["macro_f1"].append(f1_score(y[te], pred, average="macro"))
    return data


def run_survival():
    df = pd.read_csv(SUR_DIR / "mRNA_Survival_final.tsv", sep="\t")
    y_event = df["os_event"].astype(int).values
    y_time = df["os_time_days"].astype(float).values
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    data = {name: {"roc_auc": [], "c_index": []} for name, _ in variants_for("Survival")}
    images = {name: np.load(path) for name, path in variants_for("Survival")}
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        for name, imgs in images.items():
            prob = train_survival(imgs[tr], y_event[tr], imgs[te])
            data[name]["roc_auc"].append(roc_auc_score(y_event[te], prob))
            data[name]["c_index"].append(concordance_index(y_time[te], -prob, y_event[te]))
    return data


def safe_wilcoxon(a, b):
    try:
        return float(wilcoxon(a, b, zero_method="wilcox", correction=False).pvalue)
    except Exception:
        return float("nan")


def paired_rows(task, metric, data):
    names = list(data.keys())
    rows = []
    for a, b in combinations(names, 2):
        va = np.asarray(data[a][metric], dtype=float)
        vb = np.asarray(data[b][metric], dtype=float)
        t = ttest_rel(va, vb)
        rows.append({
            "task": task,
            "metric": metric,
            "variant_a": a,
            "variant_b": b,
            "mean_diff_a_minus_b": round(float(np.mean(va - vb)), 4),
            "t_statistic": round(float(t.statistic), 4),
            "t_pvalue": round(float(t.pvalue), 4),
            "wilcoxon_pvalue": round(safe_wilcoxon(va, vb), 4),
        })
    return rows


def main():
    pam = run_pam50()
    sur = run_survival()
    rows = []
    rows += paired_rows("PAM50", "accuracy", pam)
    rows += paired_rows("PAM50", "macro_f1", pam)
    rows += paired_rows("Survival", "roc_auc", sur)
    rows += paired_rows("Survival", "c_index", sur)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "metric", "variant_a", "variant_b", "mean_diff_a_minus_b", "t_statistic", "t_pvalue", "wilcoxon_pvalue"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}")
    for r in rows:
        print(r["task"], r["metric"], r["variant_a"], "vs", r["variant_b"], "diff", r["mean_diff_a_minus_b"], "p", r["t_pvalue"])


if __name__ == "__main__":
    main()
