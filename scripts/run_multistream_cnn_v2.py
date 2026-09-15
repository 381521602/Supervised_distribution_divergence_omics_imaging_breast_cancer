#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-stream complex residual CNN fusion for omics images."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import feature_order, to_images_with_order, spiral_order  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "multistream_cnn_v2_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 80
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
SIZES = {"mRNA": 15, "CNV": 8, "miRNA": 15}
ORDERINGS = {"mRNA": "NSRE", "CNV": "mean", "miRNA": "NSRE"}


def mean_order(X: np.ndarray) -> np.ndarray:
    return np.argsort(np.mean(X, axis=0))[::-1]


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + identity)


class ResBranch(nn.Module):
    def __init__(self, channels: int = 32, blocks: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x)


class MultiStreamResCNN(nn.Module):
    def __init__(self, out_dim: int, channels: int = 32):
        super().__init__()
        self.mrna = ResBranch(channels)
        self.cnv = ResBranch(channels)
        self.mirna = ResBranch(channels)
        self.head = nn.Sequential(
            nn.Linear(channels * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x_mrna, x_cnv, x_mirna):
        f1 = self.mrna(x_mrna)
        f2 = self.cnv(x_cnv)
        f3 = self.mirna(x_mirna)
        return self.head(torch.cat([f1, f2, f3], dim=1))


def torch_seed():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def run_fold(train_images, test_images, y_train, y_test, out_dim):
    torch_seed()
    model = MultiStreamResCNN(out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    X_train = [torch.tensor(x, dtype=torch.float32) for x in train_images]
    yt = torch.tensor(y_train, dtype=torch.long)
    model.train()
    n = X_train[0].shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            optimizer.zero_grad()
            out = model(X_train[0][idx], X_train[1][idx], X_train[2][idx])
            loss = criterion(out, yt[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    X_test = [torch.tensor(x, dtype=torch.float32) for x in test_images]
    with torch.no_grad():
        logits = model(X_test[0], X_test[1], X_test[2])
        pred = logits.argmax(dim=1).numpy()
    return pred


def make_images(block, y_train, omics):
    scaler = StandardScaler().fit(block[y_train == y_train])
    # scaler fitted above is not used directly; caller handles train/test separately.


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    matrices = [pd.read_csv(FEAT_DIR / f"{omics}_PAM50_4class_matrix.tsv", sep="\t", index_col=0) for omics in OMICS]
    common = matrices[0].index
    for m in matrices[1:]:
        common = common.intersection(m.index)
    common = list(common)
    X_blocks = [m.loc[common].to_numpy(dtype=float) for m in matrices]
    y_raw = labels.set_index("case_id").loc[common, "pam50_4class"].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for train_idx, test_idx in cv.split(np.zeros(len(common)), y):
        train_images, test_images = [], []
        for block, omics in zip(X_blocks, OMICS):
            scaler = StandardScaler().fit(block[train_idx])
            X_train = scaler.transform(block[train_idx])
            X_test = scaler.transform(block[test_idx])
            if ORDERINGS[omics] == "mean":
                order = mean_order(X_train)
            else:
                order = feature_order(X_train, y[train_idx])
            train_images.append(to_images_with_order(X_train, SIZES[omics], order))
            test_images.append(to_images_with_order(X_test, SIZES[omics], order))
        pred = run_fold(train_images, test_images, y[train_idx], y[test_idx], len(encoder.classes_))
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))

    rows = [
        {"model": "MultiStreamResCNN", "metric": "accuracy", "value": round(float(np.mean(accs)), 4)},
        {"model": "MultiStreamResCNN", "metric": "macro_f1", "value": round(float(np.mean(f1s)), 4)},
    ]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Accuracy={rows[0]['value']} Macro-F1={rows[1]['value']}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
