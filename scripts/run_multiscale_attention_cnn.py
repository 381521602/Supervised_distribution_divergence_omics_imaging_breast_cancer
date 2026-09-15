#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-scale full-size conv + pointwise + SE + cross-omics attention."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import feature_order, to_images_with_order  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "multiscale_attention_cnn_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 80
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
SIZES = {"mRNA": 15, "CNV": 8, "miRNA": 15}
SCALES = {"mRNA": [15, 7, 3], "CNV": [8, 4], "miRNA": [15, 7, 3]}
CHANNELS = 32


class ScaleBlock(nn.Module):
    def __init__(self, kernel: int, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size=(kernel, kernel), padding=0)
        self.bn = nn.BatchNorm2d(channels)
        self.point = nn.Conv2d(channels, channels, 1)
        self.point_bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.point(x)
        x = F.relu(self.point_bn(x))
        return x.flatten(1)


class SE(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, dim // reduction)
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class Branch(nn.Module):
    def __init__(self, scales: list[int], channels: int):
        super().__init__()
        self.blocks = nn.ModuleList([ScaleBlock(k, channels) for k in scales])
        self.se = SE(channels * len(scales))

    def forward(self, x):
        features = [block(x) for block in self.blocks]
        out = torch.cat(features, dim=1)
        return self.se(out)


class MultiScaleAttentionNet(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.mrna = Branch(SCALES["mRNA"], CHANNELS)
        self.cnv = Branch(SCALES["CNV"], CHANNELS)
        self.mirna = Branch(SCALES["miRNA"], CHANNELS)
        self.mrna_dim = CHANNELS * len(SCALES["mRNA"])
        self.cnv_dim = CHANNELS * len(SCALES["CNV"])
        self.mirna_dim = CHANNELS * len(SCALES["miRNA"])
        total = self.mrna_dim + self.cnv_dim + self.mirna_dim
        self.attn = nn.Linear(total, 3)
        self.head = nn.Sequential(
            nn.Linear(total, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x_mrna, x_cnv, x_mirna):
        v1 = self.mrna(x_mrna)
        v2 = self.cnv(x_cnv)
        v3 = self.mirna(x_mirna)
        concat = torch.cat([v1, v2, v3], dim=1)
        weights = torch.softmax(self.attn(concat), dim=1)
        weighted = torch.cat(
            [weights[:, 0:1] * v1, weights[:, 1:2] * v2, weights[:, 2:3] * v3],
            dim=1,
        )
        return self.head(weighted)


def torch_seed():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def run_fold(train_imgs, test_imgs, y_train, y_test, out_dim):
    torch_seed()
    model = MultiScaleAttentionNet(out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    X_train = [torch.tensor(x, dtype=torch.float32) for x in train_imgs]
    X_test = [torch.tensor(x, dtype=torch.float32) for x in test_imgs]
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
            loss = criterion(model(X_train[0][idx], X_train[1][idx], X_train[2][idx]), yt[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(X_test[0], X_test[1], X_test[2])
    return logits.argmax(dim=1).numpy()


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
        train_imgs, test_imgs = [], []
        for block, omics in zip(X_blocks, OMICS):
            scaler = StandardScaler().fit(block[train_idx])
            X_train = scaler.transform(block[train_idx])
            X_test = scaler.transform(block[test_idx])
            order = feature_order(X_train, y[train_idx])
            train_imgs.append(to_images_with_order(X_train, SIZES[omics], order))
            test_imgs.append(to_images_with_order(X_test, SIZES[omics], order))
        pred = run_fold(train_imgs, test_imgs, y[train_idx], y[test_idx], len(encoder.classes_))
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))

    rows = [
        {"model": "MultiScaleAttentionNet", "metric": "accuracy", "value": round(float(np.mean(accs)), 4)},
        {"model": "MultiScaleAttentionNet", "metric": "macro_f1", "value": round(float(np.mean(f1s)), 4)},
    ]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Accuracy={rows[0]['value']} Macro-F1={rows[1]['value']}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
