#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Category-masking interpretability experiment for CNV and miRNA."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
IMG = ROOT / "data/images"
PAM_DIR = ROOT / "data/final_datasets/PAM50"
SUR_DIR = ROOT / "data/final_datasets/Survival"
OUT = ROOT / "data/other_omics_category_masking_results.tsv"
FIG = ROOT / "data/interpretability/figures/fig_other_omics_category_masking.png"

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


def variants(images, grid):
    size = images.shape[2]
    out = {"baseline": images}
    for c in sorted(np.unique(grid).tolist()):
        x = images.copy()
        x[:, 0, grid == c] = 0.0
        out[f"mask_category_{c}"] = x
    return out, size


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


def run_omics(task, omics):
    if task == "PAM50":
        df = pd.read_csv(PAM_DIR / f"{omics}_PAM50_final.tsv", sep="\t")
        enc = LabelEncoder()
        y = enc.fit_transform(df["pam50"].values)
        metric_pairs = [("accuracy", accuracy_score), ("macro_f1", lambda yt, yp: f1_score(yt, yp, average="macro"))]
    else:
        df = pd.read_csv(SUR_DIR / f"{omics}_Survival_final.tsv", sep="\t")
        y = df["os_event"].astype(int).values
        y_time = df["os_time_days"].astype(float).values
        metric_pairs = []
    images = np.load(IMG / task / omics / "images.npy").astype(np.float32)
    grid = np.load(IMG / task / omics / "category_grid.npy").astype(int)
    vars_dict, size = variants(images, grid)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metrics = {name: [] for name in vars_dict}
    if task == "PAM50":
        for tr, te in cv.split(np.zeros(len(y)), y):
            for name, x in vars_dict.items():
                pred = train_pam50(x[tr], y[tr], x[te], size)
                metrics[name].append({m: fn(y[te], pred) for m, fn in metric_pairs})
    else:
        for tr, te in cv.split(np.zeros(len(y)), y):
            for name, x in vars_dict.items():
                prob = train_survival(x[tr], y[tr], x[te], size)
                metrics[name].append(
                    {
                        "roc_auc": roc_auc_score(y[te], prob),
                        "c_index": concordance_index(y_time[te], -prob, y[te]),
                    }
                )

    rows = []
    for name, vals in metrics.items():
        for metric in metrics[name][0].keys():
            arr = [v[metric] for v in vals]
            rows.append(
                {
                    "omics": omics,
                    "task": task,
                    "variant": name,
                    "category": CATEGORY_NAMES.get(int(name.rsplit("_", 1)[-1]), "Baseline") if name != "baseline" else "Baseline",
                    "metric": metric,
                    "mean": round(float(np.mean(arr)), 4),
                    "std": round(float(np.std(arr)), 4),
                }
            )
    return rows


def make_figure(rows):
    df = pd.DataFrame(rows)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    omics = ["CNV", "miRNA"]
    tasks = ["PAM50", "Survival"]
    metric_map = {"PAM50": ("accuracy", "Accuracy"), "Survival": ("c_index", "C-index")}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for i, om in enumerate(omics):
        for j, task in enumerate(tasks):
            ax = axes[j, i]
            metric, label = metric_map[task]
            sub = df[(df["omics"] == om) & (df["task"] == task) & (df["metric"] == metric)].copy()
            sub = sub.sort_values("category")
            colors = ["#2E6FA6" if r["category"] == "Baseline" else "#D86C6C" for _, r in sub.iterrows()]
            ax.bar(sub["category"], sub["mean"], yerr=sub["std"], capsize=4, color=colors, alpha=0.9)
            ax.set_title(f"{om} · {task} · {label}", fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=22, labelsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.set_ylim(0, max(sub["mean"].max() + sub["std"].max() + 0.05, 0.85))
    fig.suptitle("CNV and miRNA functional-category masking", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG, dpi=300)
    plt.close(fig)
    print(f"Saved figure -> {FIG}")


def main():
    rows = []
    for omics in ["CNV", "miRNA"]:
        rows += run_omics("PAM50", omics)
        rows += run_omics("Survival", omics)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["omics", "task", "variant", "category", "metric", "mean", "std"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    make_figure(rows)
    print(f"Wrote -> {OUT}")
    for r in rows:
        print(r["omics"], r["task"], r["variant"], r["metric"], r["mean"], "+-", r["std"])


if __name__ == "__main__":
    main()
