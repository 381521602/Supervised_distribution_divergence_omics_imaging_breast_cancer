#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NMF component activities -> NSRE-weighted images -> FullSizeCNN."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import nsre_scores, spiral_order  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "nmf_fullsize_cnn_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
NMF_K = {"mRNA": 100, "CNV": 20, "miRNA": 100}
SIZES = {"mRNA": 10, "CNV": 5, "miRNA": 10}
CHANNELS = 32


def weighted_images(X, size, order, scores):
    positions = spiral_order(size)
    n = X.shape[0]
    imgs = np.zeros((n, 1, size, size), dtype=np.float32)
    if scores.max() > scores.min():
        w = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        w = np.zeros_like(scores)
    for i in range(n):
        for pos, feat_idx in zip(positions, order):
            if feat_idx < X.shape[1]:
                imgs[i, 0, pos[0], pos[1]] = X[i, feat_idx] * w[feat_idx]
    return imgs


class ConvBranch(nn.Module):
    def __init__(self, size, channels):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size=(size, size))
        self.bn = nn.BatchNorm2d(channels)
    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.flatten(1)


class MultiStreamNMFNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mrna = ConvBranch(SIZES["mRNA"], CHANNELS)
        self.cnv = ConvBranch(SIZES["CNV"], CHANNELS)
        self.mirna = ConvBranch(SIZES["miRNA"], CHANNELS)
        self.head = nn.Sequential(
            nn.Linear(CHANNELS * 3, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )
    def forward(self, x1, x2, x3):
        return self.head(torch.cat([self.mrna(x1), self.cnv(x2), self.mirna(x3)], dim=1))


def torch_seed():
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)


def run_fold(train_imgs, test_imgs, y_train, y_test, out_dim):
    torch_seed()
    model = MultiStreamNMFNet(out_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    X_train = [torch.tensor(x, dtype=torch.float32) for x in train_imgs]
    X_test = [torch.tensor(x, dtype=torch.float32) for x in test_imgs]
    yt = torch.tensor(y_train, dtype=torch.long)
    n = X_train[0].shape[0]; model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            if idx.shape[0] < 2: continue
            opt.zero_grad(); loss = crit(model(X_train[0][idx], X_train[1][idx], X_train[2][idx]), yt[idx]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(X_test[0], X_test[1], X_test[2])
    return logits.argmax(dim=1).numpy()


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    matrices = [pd.read_csv(FEAT_DIR / f"{omics}_PAM50_4class_matrix.tsv", sep="\t", index_col=0) for omics in OMICS]
    common = matrices[0].index
    for m in matrices[1:]: common = common.intersection(m.index)
    common = list(common)
    X_blocks = [m.loc[common].to_numpy(dtype=float) for m in matrices]
    y_raw = labels.set_index("case_id").loc[common, "pam50_4class"].values
    enc = LabelEncoder(); y = enc.fit_transform(y_raw)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, te in cv.split(np.zeros(len(common)), y):
        train_imgs, test_imgs = [], []
        for block, omics in zip(X_blocks, OMICS):
            scaler = MinMaxScaler(clip=True).fit(block[tr])
            Xtr_nn = scaler.transform(block[tr]); Xte_nn = scaler.transform(block[te])
            nmf = NMF(n_components=NMF_K[omics], init="nndsvda", random_state=RANDOM_STATE, max_iter=500)
            W_train = nmf.fit_transform(Xtr_nn)
            W_test = nmf.transform(Xte_nn)
            scores = nsre_scores(W_train, y[tr]); order = np.argsort(scores)[::-1]
            train_imgs.append(weighted_images(W_train, SIZES[omics], order, scores))
            test_imgs.append(weighted_images(W_test, SIZES[omics], order, scores))
        pred = run_fold(train_imgs, test_imgs, y[tr], y[te], len(enc.classes_))
        accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
    rows = [
        {"model": "NMF_NSRE_FullSizeCNN", "metric": "accuracy", "value": round(float(np.mean(accs)), 4)},
        {"model": "NMF_NSRE_FullSizeCNN", "metric": "macro_f1", "value": round(float(np.mean(f1s)), 4)},
    ]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"Accuracy={rows[0]['value']} Macro-F1={rows[1]['value']}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
