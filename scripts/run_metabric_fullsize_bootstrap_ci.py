# -*- coding: utf-8 -*-
"""Bootstrap 95% CIs for the METABRIC external-validation models.

Re-runs the 342-shared-gene mRNA PAM50 external validation (same pipeline as
run_metabric_fullsize_external.py), saves per-sample predictions, and computes
percentile bootstrap confidence intervals for Accuracy, Macro-F1, and the
paired differences FullSizeCNN - Dense.
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_PRED = DATA / "results" / "metabric_fullsize_per_sample_predictions.tsv"
OUT_CI = DATA / "results" / "metabric_fullsize_bootstrap_ci.tsv"
MATRIX = DATA / "external" / "metabric" / "metabric_mrna_zscores_selected.tsv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_metabric_external_validation import (  # noqa: E402
    load_tcga_pam50,
    load_metabric_clinical,
    prepare_metabric_labels,
)
from run_multistream_cnn import jsd_scores, spiral_order  # noqa: E402

RANDOM_STATE = 42
EPOCHS = 30
BATCH = 64
N_BOOT = 2000
CLASSES = ["Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]


class FullSizeCNN(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.head(F.relu(self.conv(x)))


class Dense(nn.Module):
    def __init__(self, din, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_torch(model, Xtr, ytr, Xte):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    crit = nn.CrossEntropyLoss()
    n = yt.shape[0]
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad()
            crit(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte, dtype=torch.float32))
        return logits.argmax(1).numpy(), F.softmax(logits, dim=1).numpy()


def make_images(X, order, size):
    positions = spiral_order(size)
    imgs = np.zeros((X.shape[0], 1, size, size), dtype=np.float32)
    for i in range(X.shape[0]):
        for (r, c), j in zip(positions, order):
            if j < X.shape[1]:
                imgs[i, 0, r, c] = X[i, j]
    return imgs


def bootstrap_ci(y_true, preds: dict[str, np.ndarray], n_boot=N_BOOT, seed=42):
    """Percentile bootstrap CIs for Accuracy/Macro-F1 and paired model diffs."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    models = list(preds.keys())

    def acc(yt, yp):
        return float(accuracy_score(yt, yp))

    def mf1(yt, yp):
        return float(f1_score(yt, yp, average="macro"))

    boot = {m: {"accuracy": [], "macro_f1": []} for m in models}
    diff = {(a, b): {"accuracy": [], "macro_f1": []} for a in models for b in models if a < b}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        per = {}
        for m in models:
            yb = preds[m][idx]
            per[m] = {"accuracy": acc(yt, yb), "macro_f1": mf1(yt, yb)}
            for k in ("accuracy", "macro_f1"):
                boot[m][k].append(per[m][k])
        for (a, b) in diff:
            for k in ("accuracy", "macro_f1"):
                diff[(a, b)][k].append(per[a][k] - per[b][k])

    rows = []
    for m in models:
        for k in ("accuracy", "macro_f1"):
            arr = np.array(boot[m][k])
            fn = acc if k == "accuracy" else mf1
            point = fn(y_true, preds[m])
            rows.append(
                {
                    "comparison": m,
                    "metric": k,
                    "point": round(point, 4),
                    "lo95": round(float(np.percentile(arr, 2.5)), 4),
                    "hi95": round(float(np.percentile(arr, 97.5)), 4),
                }
            )
    for (a, b) in diff:
        for k in ("accuracy", "macro_f1"):
            arr = np.array(diff[(a, b)][k])
            fn = acc if k == "accuracy" else mf1
            point = fn(y_true, preds[a]) - fn(y_true, preds[b])
            lo, hi = np.percentile(arr, 2.5), np.percentile(arr, 97.5)
            p_two_sided = float(min(np.mean(arr <= 0), np.mean(arr >= 0)) * 2)
            rows.append(
                {
                    "comparison": f"{a}_minus_{b}",
                    "metric": k,
                    "point": round(point, 4),
                    "lo95": round(float(lo), 4),
                    "hi95": round(float(hi), 4),
                    "p_two_sided": round(p_two_sided, 4),
                }
            )
    return rows


def main():
    matrix = pd.read_csv(MATRIX, sep="\t", index_col=0)  # genes x samples
    clinical = load_metabric_clinical()
    pam_valid, _ = prepare_metabric_labels(clinical, matrix)
    y_tcga, tcga_genes, X_tcga = load_tcga_pam50()
    common = [g for g in tcga_genes if g in matrix.index]

    Xtr_raw = X_tcga[:, [tcga_genes.index(g) for g in common]]
    Xm_raw = np.nan_to_num(matrix.loc[common].T.to_numpy(dtype=float), nan=0.0)
    Xtr = StandardScaler().fit(Xtr_raw).transform(Xtr_raw)
    Xm = StandardScaler().fit(Xm_raw).transform(Xm_raw)

    common_samples = list(matrix.columns.intersection(pam_valid.index))
    pos = {c: i for i, c in enumerate(matrix.columns)}
    Xte = Xm[[pos[c] for c in common_samples]]
    yte = pam_valid.loc[common_samples].astype(str).values
    y_tcga_enc = np.array([CLASSES.index(s) for s in y_tcga])
    yte_enc = np.array([CLASSES.index(s) for s in yte])

    order = np.argsort(jsd_scores(Xtr, y_tcga_enc))[::-1]
    size = 19  # 19*19 = 361 >= 342
    imgs_tr = make_images(Xtr[:, order], np.arange(Xtr.shape[1]), size)
    imgs_te = make_images(Xte[:, order], np.arange(Xte.shape[1]), size)

    preds: dict[str, np.ndarray] = {}
    lr = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE).fit(Xtr, y_tcga_enc)
    preds["LogisticRegression"] = lr.predict(Xte)
    preds["Dense"] = train_torch(Dense(Xtr.shape[1], 4), Xtr, y_tcga_enc, Xte)[0]
    preds["FullSizeCNN"] = train_torch(FullSizeCNN(size, 4), imgs_tr, y_tcga_enc, imgs_te)[0]

    # Save per-sample predictions.
    with OUT_PRED.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sample", "true_label", *preds.keys()])
        for i, s in enumerate(common_samples):
            w.writerow([s, yte[i], *[preds[m][i] for m in preds]])

    rows = bootstrap_ci(yte_enc, preds)
    fieldnames = ["comparison", "metric", "point", "lo95", "hi95", "p_two_sided"]
    with OUT_CI.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"TCGA features={len(tcga_genes)} shared_genes={len(common)} METABRIC_n={len(yte_enc)}")
    for r in rows:
        print(r)
    print("Wrote", OUT_PRED, "and", OUT_CI)


if __name__ == "__main__":
    main()
