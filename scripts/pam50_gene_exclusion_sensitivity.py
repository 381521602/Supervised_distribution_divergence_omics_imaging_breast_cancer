#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PAM50 50-gene exclusion sensitivity analysis on mRNA PAM50 task.

Compares three feature configurations on the mRNA PAM50 classification task:
  1. full_transcriptome : the 400 pre-selected final features (baseline)
  2. exclude_pam50_genes: the 400 features with any PAM50 50-gene overlap removed
  3. pam50_only         : only the PAM50 50 genes that are present in the 400 features

Models: LogisticRegression, MLP, FullSizeCNN.
Protocol: 5-fold stratified CV, mean +/- SD of Accuracy and Macro-F1.
Feature scaling and JSD ordering are computed inside each training fold (no leakage).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import jsd_scores, spiral_order  # noqa: E402

PAM_MRNA = ROOT / "data" / "final_datasets" / "PAM50" / "mRNA_PAM50_final.tsv"
OUT = ROOT / "data" / "pam50_gene_exclusion_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64

PAM50_GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2",
    "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C",
    "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67",
    "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6L",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]


class MLP(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, dout),
        )

    def forward(self, x):
        return self.net(x)


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


def make_images(X: np.ndarray, y: np.ndarray) -> tuple:
    n = X.shape[1]
    size = int(np.ceil(np.sqrt(n)))
    scores = jsd_scores(X, y)
    order = np.argsort(scores)[::-1]
    positions = spiral_order(size)
    imgs = np.zeros((X.shape[0], 1, size, size), dtype=np.float32)
    for i in range(X.shape[0]):
        for pos, feat_idx in zip(positions, order):
            if feat_idx < n:
                imgs[i, 0, pos[0], pos[1]] = X[i, feat_idx]
    return imgs, size


def train_mlp(Xtr, ytr, Xte):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = MLP(Xtr.shape[1], 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    crit = nn.CrossEntropyLoss()
    n = yt.shape[0]
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
    return F.softmax(out, dim=1).numpy()


def train_cnn(imgs_tr, ytr, imgs_te, size):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeCNN(size, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(imgs_tr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    crit = nn.CrossEntropyLoss()
    n = yt.shape[0]
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
        out = model(torch.tensor(imgs_te, dtype=torch.float32))
    return F.softmax(out, dim=1).numpy()


def eval_config(config, X, y, imgs, size):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {"LogisticRegression": {"acc": [], "f1": []}, "MLP": {"acc": [], "f1": []}, "FullSizeCNN": {"acc": [], "f1": []}}
    for tr, te in cv.split(np.zeros(len(y)), y):
        pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        results["LogisticRegression"]["acc"].append(accuracy_score(y[te], pred))
        results["LogisticRegression"]["f1"].append(f1_score(y[te], pred, average="macro"))

        p = train_mlp(X[tr], y[tr], X[te])
        pred = p.argmax(axis=1)
        results["MLP"]["acc"].append(accuracy_score(y[te], pred))
        results["MLP"]["f1"].append(f1_score(y[te], pred, average="macro"))

        p = train_cnn(imgs[tr], y[tr], imgs[te], size)
        pred = p.argmax(axis=1)
        results["FullSizeCNN"]["acc"].append(accuracy_score(y[te], pred))
        results["FullSizeCNN"]["f1"].append(f1_score(y[te], pred, average="macro"))

    rows = []
    for name, m in results.items():
        rows.append({"config": config, "model": name, "metric": "accuracy", "mean": round(float(np.mean(m["acc"])), 4), "std": round(float(np.std(m["acc"])), 4)})
        rows.append({"config": config, "model": name, "metric": "macro_f1", "mean": round(float(np.mean(m["f1"])), 4), "std": round(float(np.std(m["f1"])), 4)})
    return rows


def main():
    df = pd.read_csv(PAM_MRNA, sep="\t")
    features = list(df.columns[1:])
    X_all = df[features].to_numpy(dtype=float)
    enc = LabelEncoder()
    y = enc.fit_transform(df["pam50"].values)

    overlap = [g for g in PAM50_GENES if g in features]
    print(f"400 features, PAM50-gene overlap = {len(overlap)}: {overlap}", flush=True)

    configs = {}
    configs["full_transcriptome"] = features
    configs["exclude_pam50_genes"] = [f for f in features if f not in overlap]
    configs["pam50_only"] = overlap

    rows = []
    for config, cols in configs.items():
        X = df[cols].to_numpy(dtype=float)
        imgs, size = make_images(X, y)
        print(f"{config}: n_features={X.shape[1]}, img_size={size}x{size}", flush=True)
        rows += eval_config(config, X, y, imgs, size)
        print(f"done {config}", flush=True)

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "model", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
