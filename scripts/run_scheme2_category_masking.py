#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheme-2 functional-category masking interpretability experiment.

For each mRNA functional category in the scheme-2 center-importance image,
we zero out the pixels belonging to that category, train a simple
FullSizeCNN, and measure the resulting performance drop on PAM50 subtyping
and overall-survival prediction.
"""

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
PAM_DIR = ROOT / "data/final_datasets/PAM50"
SUR_DIR = ROOT / "data/final_datasets/Survival"
IMG = ROOT / "data/images"
OUT = ROOT / "data/scheme2_category_masking_results.tsv"
FIG = ROOT / "data/interpretability/figures/fig_scheme2_category_masking.png"

RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 24
BATCH_SIZE = 64

CATEGORY_NAMES = {
    0: "Development / Epithelium",
    1: "Signaling / Transport",
    2: "Hormone / Metabolism",
    3: "Immune / Inflammation",
    4: "Cell cycle / Proliferation",
    5: "Other",
}


class FullSizeCNN(nn.Module):
    def __init__(self, size: int, out_dim: int):
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


def image_variants(task: str):
    reorder = IMG / task / "mRNA_reorder/scheme2_function_center"
    gray = np.load(reorder / "images.npy").astype(np.float32)
    cat_grid = np.load(reorder / "category_grid.npy").astype(np.int64)
    size = gray.shape[2]
    variants = {"baseline": gray}
    for c in sorted(np.unique(cat_grid).tolist()):
        masked = gray.copy()
        masked[:, 0, cat_grid == c] = 0.0
        variants[f"mask_category_{c}"] = masked
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


def run_pam50():
    df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
    enc = LabelEncoder()
    y = enc.fit_transform(df["pam50"].values)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    variants, size = image_variants("PAM50")
    metrics = {name: {"acc": [], "f1": []} for name in variants}
    for tr, te in cv.split(np.zeros(len(y)), y):
        for name, x in variants.items():
            pred = train_pam50(x[tr], y[tr], x[te], size)
            metrics[name]["acc"].append(accuracy_score(y[te], pred))
            metrics[name]["f1"].append(f1_score(y[te], pred, average="macro"))
    rows = []
    for name, m in metrics.items():
        rows.append({"variant": name, "category": _category_for(name), "task": "PAM50", "metric": "accuracy", "mean": round(float(np.mean(m["acc"])), 4), "std": round(float(np.std(m["acc"])), 4)})
        rows.append({"variant": name, "category": _category_for(name), "task": "PAM50", "metric": "macro_f1", "mean": round(float(np.mean(m["f1"])), 4), "std": round(float(np.std(m["f1"])), 4)})
    return rows


def run_survival():
    df = pd.read_csv(SUR_DIR / "mRNA_Survival_final.tsv", sep="\t")
    y_event = df["os_event"].astype(int).values
    y_time = df["os_time_days"].astype(float).values
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    variants, size = image_variants("Survival")
    metrics = {name: {"auc": [], "ci": []} for name in variants}
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        for name, x in variants.items():
            prob = train_survival(x[tr], y_event[tr], x[te], size)
            metrics[name]["auc"].append(roc_auc_score(y_event[te], prob))
            metrics[name]["ci"].append(concordance_index(y_time[te], -prob, y_event[te]))
    rows = []
    for name, m in metrics.items():
        rows.append({"variant": name, "category": _category_for(name), "task": "Survival", "metric": "roc_auc", "mean": round(float(np.mean(m["auc"])), 4), "std": round(float(np.std(m["auc"])), 4)})
        rows.append({"variant": name, "category": _category_for(name), "task": "Survival", "metric": "c_index", "mean": round(float(np.mean(m["ci"])), 4), "std": round(float(np.std(m["ci"])), 4)})
    return rows


def _category_for(name: str) -> str:
    if name == "baseline":
        return "Baseline"
    return CATEGORY_NAMES.get(int(name.rsplit("_", 1)[-1]), name)


def make_figure(rows):
    df = pd.DataFrame(rows)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    tasks = [("PAM50", "accuracy", "Accuracy"), ("Survival", "c_index", "C-index")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (task, metric, label) in zip(axes, tasks):
        sub = df[(df["task"] == task) & (df["metric"] == metric)].copy()
        sub["category"] = pd.Categorical(sub["category"], categories=sub["category"].tolist(), ordered=True)
        sub = sub.sort_values("category")
        colors = ["#2E6FA6" if r["category"] == "Baseline" else "#D86C6C" for _, r in sub.iterrows()]
        ax.bar(sub["category"], sub["mean"], yerr=sub["std"], capsize=4, color=colors, alpha=0.9)
        ax.set_title(f"{task} · {label}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(sub["mean"].max() + sub["std"].max() + 0.04, 0.85))
        ax.tick_params(axis="x", rotation=22, labelsize=8.5)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle("Scheme-2 mRNA functional-category masking", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG, dpi=300)
    plt.close(fig)
    print(f"Saved figure -> {FIG}")


def main():
    rows = run_pam50() + run_survival()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "category", "task", "metric", "mean", "std"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    make_figure(rows)
    print(f"Wrote -> {OUT}")
    for r in rows:
        print(r["variant"], "|", r["category"], "|", r["task"], "|", r["metric"], "|", r["mean"], "+-", r["std"])


if __name__ == "__main__":
    main()
