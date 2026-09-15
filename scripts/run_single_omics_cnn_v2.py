#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Improved single-omics CNN tailored to small grayscale feature maps."""

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
from run_multistream_cnn import feature_order, to_images_with_order  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "single_omics_cnn_v2_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 80
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
IMAGE_SIZE = {"mRNA": 15, "CNV": 8, "miRNA": 15}


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
        out = self.relu(out + identity)
        return out


class SmallResCNN(nn.Module):
    def __init__(self, out_dim: int, channels: int = 32, blocks: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def torch_seed():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def run_fold(X_train, X_test, y_train, y_test, out_dim):
    torch_seed()
    model = SmallResCNN(out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    model.train()
    n = Xt.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            optimizer.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        pred = logits.argmax(dim=1).numpy()
    return pred


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    rows: list[dict] = []

    for omics in OMICS:
        matrix = pd.read_csv(FEAT_DIR / f"{omics}_PAM50_4class_matrix.tsv", sep="\t", index_col=0)
        common = list(matrix.index)
        X = matrix.loc[common].to_numpy(dtype=float)
        y_raw = labels.set_index("case_id").loc[common, "pam50_4class"].values
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw)
        size = IMAGE_SIZE[omics]

        cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        accs, f1s = [], []
        for train_idx, test_idx in cv.split(np.zeros(len(common)), y):
            scaler = StandardScaler().fit(X[train_idx])
            X_train = scaler.transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            order = feature_order(X_train, y[train_idx])
            X_train_img = to_images_with_order(X_train, size, order)
            X_test_img = to_images_with_order(X_test, size, order)
            pred = run_fold(X_train_img, X_test_img, y[train_idx], y[test_idx], len(encoder.classes_))
            accs.append(accuracy_score(y[test_idx], pred))
            f1s.append(f1_score(y[test_idx], pred, average="macro"))

        rows.append({"omics": omics, "metric": "accuracy", "value": round(float(np.mean(accs)), 4)})
        rows.append({"omics": omics, "metric": "macro_f1", "value": round(float(np.mean(f1s)), 4)})
        print(f"{omics}: accuracy={rows[-2]['value']} macro_f1={rows[-1]['value']}")

    if rows:
        columns = ["omics", "metric", "value"]
        with OUT.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
