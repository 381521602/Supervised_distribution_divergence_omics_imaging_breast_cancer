#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dual-path model: full-size conv image path + raw-vector MLP path."""

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
OUT = DATA / "dual_path_cnn_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 80
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
SIZES = {"mRNA": 15, "CNV": 8, "miRNA": 15}
CHANNELS = 32


class ConvBranch(nn.Module):
    def __init__(self, size: int, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size=(size, size), padding=0)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.flatten(1)


class ImagePath(nn.Module):
    def __init__(self):
        super().__init__()
        self.mrna = ConvBranch(SIZES["mRNA"], CHANNELS)
        self.cnv = ConvBranch(SIZES["CNV"], CHANNELS)
        self.mirna = ConvBranch(SIZES["miRNA"], CHANNELS)

    def forward(self, x_mrna, x_cnv, x_mirna):
        f1 = self.mrna(x_mrna)
        f2 = self.cnv(x_cnv)
        f3 = self.mirna(x_mirna)
        return torch.cat([f1, f2, f3], dim=1)


class VectorPath(nn.Module):
    def __init__(self, raw_dim: int, out_dim: int = 96):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.mlp(x)


class DualPathNet(nn.Module):
    def __init__(self, out_dim: int, raw_dim: int):
        super().__init__()
        self.image_path = ImagePath()
        self.vector_path = VectorPath(raw_dim)
        image_dim = CHANNELS * 3
        vector_dim = 96
        self.head = nn.Sequential(
            nn.Linear(image_dim + vector_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x_mrna_img, x_cnv_img, x_mirna_img, raw):
        img_feat = self.image_path(x_mrna_img, x_cnv_img, x_mirna_img)
        vec_feat = self.vector_path(raw)
        return self.head(torch.cat([img_feat, vec_feat], dim=1))


def torch_seed():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def run_fold(train_imgs, test_imgs, raw_train, raw_test, y_train, y_test, out_dim):
    torch_seed()
    model = DualPathNet(out_dim, raw_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    X_train = [torch.tensor(x, dtype=torch.float32) for x in train_imgs]
    X_test = [torch.tensor(x, dtype=torch.float32) for x in test_imgs]
    R_train = torch.tensor(raw_train, dtype=torch.float32)
    R_test = torch.tensor(raw_test, dtype=torch.float32)
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
            out = model(X_train[0][idx], X_train[1][idx], X_train[2][idx], R_train[idx])
            loss = criterion(out, yt[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(X_test[0], X_test[1], X_test[2], R_test)
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
        raw_parts_train, raw_parts_test = [], []
        for block, omics in zip(X_blocks, OMICS):
            scaler = StandardScaler().fit(block[train_idx])
            X_train = scaler.transform(block[train_idx])
            X_test = scaler.transform(block[test_idx])
            raw_parts_train.append(X_train)
            raw_parts_test.append(X_test)
            order = feature_order(X_train, y[train_idx])
            train_imgs.append(to_images_with_order(X_train, SIZES[omics], order))
            test_imgs.append(to_images_with_order(X_test, SIZES[omics], order))
        raw_train = np.hstack(raw_parts_train)
        raw_test = np.hstack(raw_parts_test)
        pred = run_fold(train_imgs, test_imgs, raw_train, raw_test, y[train_idx], y[test_idx], len(encoder.classes_))
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))

    rows = [
        {"model": "DualPath_Image_CNN_Vector_MLP", "metric": "accuracy", "value": round(float(np.mean(accs)), 4)},
        {"model": "DualPath_Image_CNN_Vector_MLP", "metric": "macro_f1", "value": round(float(np.mean(f1s)), 4)},
    ]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Accuracy={rows[0]['value']} Macro-F1={rows[1]['value']}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
