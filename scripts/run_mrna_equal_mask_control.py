#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Equal-size random masking control for mRNA scheme-2 category masks."""

from __future__ import annotations

import csv
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
IMG = ROOT / "data/images"
PAM_DIR = ROOT / "data/final_datasets/PAM50"
SUR_DIR = ROOT / "data/final_datasets/Survival"
OUT_FOLDS = ROOT / "data/mrna_equal_mask_control_folds.tsv"
OUT_TESTS = ROOT / "data/mrna_equal_mask_control_paired_tests.tsv"

RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 64

CATEGORY_NAMES = {
    0: "Development/Epithelium",
    1: "Signaling/Transport",
    2: "Hormone/Metabolism",
    3: "Immune/Inflammation",
    4: "Cell cycle/Proliferation",
    5: "Other",
}


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


def make_variants(images, grid, task):
    size = images.shape[2]
    rng = np.random.RandomState(RANDOM_STATE)
    all_positions = [(r, c) for r in range(size) for c in range(size)]
    variants = {"baseline": images}
    for c in sorted(np.unique(grid).tolist()):
        cat_positions = list(zip(*np.where(grid == c)))
        k = len(cat_positions)
        x_cat = images.copy()
        for r, cpos in cat_positions:
            x_cat[:, 0, r, cpos] = 0.0
        variants[f"mask_category_{c}"] = x_cat
        chosen = rng.choice(len(all_positions), size=k, replace=False)
        x_rand = images.copy()
        for idx in chosen:
            r, cpos = all_positions[idx]
            x_rand[:, 0, r, cpos] = 0.0
        variants[f"random_equal_{c}"] = x_rand
    return variants, size


def train_pam50(Xtr, ytr, Xte, size):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeCNN(size, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    n = Xt.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte, dtype=torch.float32))
    return logits.argmax(dim=1).numpy()


def train_survival(Xtr, ytr, Xte, size):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeCNN(size, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    n = Xt.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte, dtype=torch.float32))
    return torch.sigmoid(logits).numpy().ravel()


def run_task(task):
    if task == "PAM50":
        df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
        enc = LabelEncoder()
        y = enc.fit_transform(df["pam50"].values)
        y_time = None
    else:
        df = pd.read_csv(SUR_DIR / "mRNA_Survival_final.tsv", sep="\t")
        y = df["os_event"].astype(int).values
        y_time = df["os_time_days"].astype(float).values
    images = np.load(IMG / task / "mRNA_reorder/scheme2_function_center/images.npy").astype(np.float32)
    grid = np.load(IMG / task / "mRNA_reorder/scheme2_function_center/category_grid.npy").astype(int)
    variants, size = make_variants(images, grid, task)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    fold_values = {name: [] for name in variants}
    if task == "PAM50":
        for fold_idx, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
            for name, x in variants.items():
                pred = train_pam50(x[tr], y[tr], x[te], size)
                val = {
                    "accuracy": accuracy_score(y[te], pred),
                    "macro_f1": f1_score(y[te], pred, average="macro"),
                }
                fold_values[name].append(val)
                fold_rows.append({"task": task, "fold": fold_idx, "variant": name, "metric": "accuracy", "value": val["accuracy"]})
                fold_rows.append({"task": task, "fold": fold_idx, "variant": name, "metric": "macro_f1", "value": val["macro_f1"]})
    else:
        for fold_idx, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
            for name, x in variants.items():
                prob = train_survival(x[tr], y[tr], x[te], size)
                val = {
                    "roc_auc": roc_auc_score(y[te], prob),
                    "c_index": concordance_index(y_time[te], -prob, y[te]),
                }
                fold_values[name].append(val)
                fold_rows.append({"task": task, "fold": fold_idx, "variant": name, "metric": "roc_auc", "value": val["roc_auc"]})
                fold_rows.append({"task": task, "fold": fold_idx, "variant": name, "metric": "c_index", "value": val["c_index"]})
    tests = []
    baseline = fold_values["baseline"]
    for name, vals in fold_values.items():
        if name == "baseline":
            continue
        for metric in baseline[0].keys():
            a = [v[metric] for v in baseline]
            b = [v[metric] for v in vals]
            t, p_t = ttest_rel(a, b)
            try:
                w, p_w = wilcoxon(a, b)
            except ValueError:
                w, p_w = np.nan, np.nan
            tests.append(
                {
                    "task": task,
                    "variant": name,
                    "metric": metric,
                    "ttest_stat": round(float(t), 4),
                    "ttest_p": round(float(p_t), 4),
                    "wilcoxon_stat": round(float(w), 4) if np.isfinite(w) else np.nan,
                    "wilcoxon_p": round(float(p_w), 4) if np.isfinite(p_w) else np.nan,
                }
            )
    return fold_rows, tests


def main():
    fold_rows, tests = [], []
    for task in ["PAM50", "Survival"]:
        fr, te = run_task(task)
        fold_rows += fr
        tests += te
    with OUT_FOLDS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "fold", "variant", "metric", "value"], delimiter="\t")
        w.writeheader()
        w.writerows(fold_rows)
    with OUT_TESTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "variant", "metric", "ttest_stat", "ttest_p", "wilcoxon_stat", "wilcoxon_p"], delimiter="\t")
        w.writeheader()
        w.writerows(tests)
    print(f"Wrote folds -> {OUT_FOLDS}")
    print(f"Wrote tests -> {OUT_TESTS}")


if __name__ == "__main__":
    main()
