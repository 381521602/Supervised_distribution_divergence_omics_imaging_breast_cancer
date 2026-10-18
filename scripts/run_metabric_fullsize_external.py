# -*- coding: utf-8 -*-
"""FullSizeCNN + controls external validation on METABRIC (mRNA PAM50)."""
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
OUT = DATA / "metabric_fullsize_external_validation_results.tsv"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_metabric_external_validation import (  # noqa: E402
    load_tcga_pam50, load_metabric_clinical, prepare_metabric_labels,
)
from run_multistream_cnn import jsd_scores, spiral_order  # noqa: E402

MATRIX = DATA / "external" / "metabric" / "metabric_mrna_zscores_selected.tsv"
RANDOM_STATE = 42
EPOCHS = 30
BATCH = 64
CLASSES = ["Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]


class FullSizeCNN(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, x):
        return self.head(F.relu(self.conv(x)))


class Dense(nn.Module):
    def __init__(self, din, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))

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
            idx = perm[i:i + BATCH]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad()
            crit(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()


def make_images(X, order, size):
    positions = spiral_order(size)
    imgs = np.zeros((X.shape[0], 1, size, size), dtype=np.float32)
    for i in range(X.shape[0]):
        for (r, c), j in zip(positions, order):
            if j < X.shape[1]:
                imgs[i, 0, r, c] = X[i, j]
    return imgs


def main():
    matrix = pd.read_csv(MATRIX, sep="\t", index_col=0)  # genes x samples
    clinical = load_metabric_clinical()
    pam_valid, _ = prepare_metabric_labels(clinical, matrix)
    y_tcga, tcga_genes, X_tcga = load_tcga_pam50()
    common = [g for g in tcga_genes if g in matrix.index]
    print("common genes", len(common), "of", len(tcga_genes))

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
    print("METABRIC test n =", len(yte_enc))

    order = np.argsort(jsd_scores(Xtr, y_tcga_enc))[::-1]
    size = 19  # 19*19 = 361 >= 342
    imgs_tr = make_images(Xtr[:, order], np.arange(Xtr.shape[1]), size)
    imgs_te = make_images(Xte[:, order], np.arange(Xte.shape[1]), size)

    rows = []
    for name, model in [("LogisticRegression", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))]:
        pred = model.fit(Xtr, y_tcga_enc).predict(Xte)
        rows.append({"model": name, "accuracy": round(float(accuracy_score(yte_enc, pred)), 4),
                     "macro_f1": round(float(f1_score(yte_enc, pred, average="macro")), 4)})
    pred = train_torch(Dense(Xtr.shape[1], 4), Xtr, y_tcga_enc, Xte)
    rows.append({"model": "Dense", "accuracy": round(float(accuracy_score(yte_enc, pred)), 4),
                 "macro_f1": round(float(f1_score(yte_enc, pred, average="macro")), 4)})
    pred = train_torch(FullSizeCNN(size, 4), imgs_tr, y_tcga_enc, imgs_te)
    rows.append({"model": "FullSizeCNN", "accuracy": round(float(accuracy_score(yte_enc, pred)), 4),
                 "macro_f1": round(float(f1_score(yte_enc, pred, average="macro")), 4)})

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "accuracy", "macro_f1"], delimiter="\t")
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
