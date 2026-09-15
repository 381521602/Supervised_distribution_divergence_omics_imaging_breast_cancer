#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regularized full-size multi-stream CNN with early stopping, scheduler, augmentation."""

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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import feature_order, to_images_with_order  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "regularized_fullsize_cnn_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 100
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
SIZES = {"mRNA": 15, "CNV": 8, "miRNA": 15}
CHANNELS = 32


class RegConvBranch(nn.Module):
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


class RegFullSizeNet(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.mrna = RegConvBranch(SIZES["mRNA"], CHANNELS)
        self.cnv = RegConvBranch(SIZES["CNV"], CHANNELS)
        self.mirna = RegConvBranch(SIZES["miRNA"], CHANNELS)
        self.head = nn.Sequential(
            nn.Linear(CHANNELS * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
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


def augment(images: list[torch.Tensor], noise_std: float = 0.02, flip_prob: float = 0.2):
    out = []
    for img in images:
        img = img.clone()
        if flip_prob > 0 and torch.rand(1).item() < flip_prob:
            img = torch.flip(img, dims=[3])
        img = img + torch.randn_like(img) * noise_std
        out.append(img)
    return out


def run_fold(train_imgs, val_imgs, test_imgs, y_train, y_val, y_test, out_dim):
    torch_seed()
    model = RegFullSizeNet(out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    X_train = [torch.tensor(x, dtype=torch.float32) for x in train_imgs]
    X_val = [torch.tensor(x, dtype=torch.float32) for x in val_imgs]
    X_test = [torch.tensor(x, dtype=torch.float32) for x in test_imgs]
    yt = torch.tensor(y_train, dtype=torch.long)
    yv = torch.tensor(y_val, dtype=torch.long)
    n = X_train[0].shape[0]
    best_val_loss = float("inf")
    best_state = None
    patience = 10
    bad_epochs = 0

    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            batch = [x[idx] for x in X_train]
            batch = augment(batch)
            optimizer.zero_grad()
            out = model(batch[0], batch[1], batch[2])
            loss = criterion(out, yt[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val[0], X_val[1], X_val[2])
            val_loss = criterion(val_out, yv).item()
        scheduler.step(val_loss)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
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
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=y[train_idx], random_state=RANDOM_STATE)

        def make_images(idx):
            imgs = []
            for block, omics in zip(X_blocks, OMICS):
                scaler = StandardScaler().fit(block[train_idx])
                X_sub = scaler.transform(block[idx])
                order = feature_order(scaler.transform(block[train_idx]), y[train_idx])
                imgs.append(to_images_with_order(X_sub, SIZES[omics], order))
            return imgs

        train_imgs = make_images(train_idx)
        val_imgs = make_images(val_idx)
        test_imgs = make_images(test_idx)
        pred = run_fold(train_imgs, val_imgs, test_imgs, y[train_idx], y[val_idx], y[test_idx], len(encoder.classes_))
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))

    rows = [
        {"model": "RegularizedFullSizeCNN", "metric": "accuracy", "value": round(float(np.mean(accs)), 4)},
        {"model": "RegularizedFullSizeCNN", "metric": "macro_f1", "value": round(float(np.mean(f1s)), 4)},
    ]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Accuracy={rows[0]['value']} Macro-F1={rows[1]['value']}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
