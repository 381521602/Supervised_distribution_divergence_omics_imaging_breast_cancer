#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-stream grayscale omics images + NSRE ordering + simple CNN baseline."""

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
from adaptive_nsre import nsre_between_histograms  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "multistream_cnn_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 50
BATCH_SIZE = 64
DEVICE = torch.device("cpu")

OMICS = ["mRNA", "CNV", "miRNA"]
IMAGE_SIZE = {"mRNA": 15, "CNV": 8, "miRNA": 15}


def load_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def nsre_scores(X: np.ndarray, y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    classes = np.unique(y)
    scores = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        feature = X[:, j]
        finite = feature[np.isfinite(feature)]
        if finite.size < 10:
            continue
        qs = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)))
        if qs.size < 2:
            continue
        bins = np.digitize(feature, qs[1:-1])
        hists = [np.bincount(bins[y == cls], minlength=n_bins) for cls in classes]
        if len(hists) == 2:
            scores[j] = nsre_between_histograms(hists[0], hists[1])
        else:
            pair_scores = [
                nsre_between_histograms(hists[a], hists[b])
                for a in range(len(hists))
                for b in range(a + 1, len(hists))
            ]
            scores[j] = float(np.mean(pair_scores)) if pair_scores else 0.0
    return scores


def spiral_order(size: int) -> list[tuple[int, int]]:
    """Center-out spiral order, highest importance placed near center."""
    positions = []
    r = c = size // 2
    positions.append((r, c))
    step = 1
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    d = 0
    while len(positions) < size * size:
        for _ in range(2):
            dr, dc = directions[d % 4]
            for _ in range(step):
                r += dr
                c += dc
                if 0 <= r < size and 0 <= c < size:
                    positions.append((r, c))
            d += 1
        step += 1
    return positions[: size * size]


def feature_order(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = nsre_scores(X, y)
    return np.argsort(scores)[::-1]


def to_images_with_order(X: np.ndarray, size: int, order: np.ndarray) -> np.ndarray:
    positions = spiral_order(size)
    n = X.shape[0]
    images = np.zeros((n, 1, size, size), dtype=np.float32)
    for i in range(n):
        img = np.zeros((size, size), dtype=np.float32)
        for pos, feat_idx in zip(positions, order):
            if feat_idx < X.shape[1]:
                img[pos] = X[i, feat_idx]
        images[i, 0] = img
    return images


class SimpleCNN(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.mrna = self._branch(1, 32)
        self.cnv = self._branch(1, 32)
        self.mirna = self._branch(1, 32)
        flat = 32 + 32 + 32
        self.head = nn.Sequential(
            nn.Linear(flat, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def _branch(self, in_ch: int, out_ch: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x_mrna, x_cnv, x_mirna):
        f1 = self.mrna(x_mrna)
        f2 = self.cnv(x_cnv)
        f3 = self.mirna(x_mirna)
        return self.head(torch.cat([f1, f2, f3], dim=1))


def torch_seed():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def run_fold(X_train_list, X_test_list, y_train, y_test, out_dim):
    torch_seed()
    model = SimpleCNN(out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_train_tensors = [torch.tensor(x, dtype=torch.float32) for x in X_train_list]
    n = X_train_tensors[0].shape[0]
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            out = model(
                X_train_tensors[0][idx],
                X_train_tensors[1][idx],
                X_train_tensors[2][idx],
            )
            loss = criterion(out, y_train_t[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    X_test_tensors = [torch.tensor(x, dtype=torch.float32) for x in X_test_list]
    with torch.no_grad():
        logits = model(X_test_tensors[0], X_test_tensors[1], X_test_tensors[2])
        pred = logits.argmax(dim=1).numpy()
    return pred


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    matrices = [load_matrix(FEAT_DIR / f"{omics}_PAM50_4class_matrix.tsv") for omics in OMICS]
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
            order = feature_order(X_train, y[train_idx])
            train_images.append(to_images_with_order(X_train, IMAGE_SIZE[omics], order))
            test_images.append(to_images_with_order(X_test, IMAGE_SIZE[omics], order))
        pred = run_fold(train_images, test_images, y[train_idx], y[test_idx], len(encoder.classes_))
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))

    rows = [
        {"model": "MultiStreamCNN_NSRE", "metric": "accuracy", "value": round(float(np.mean(accs)), 4)},
        {"model": "MultiStreamCNN_NSRE", "metric": "macro_f1", "value": round(float(np.mean(f1s)), 4)},
    ]
    columns = ["model", "metric", "value"]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {OUT}")
    print(f"Accuracy={rows[0]['value']} Macro-F1={rows[1]['value']}")


if __name__ == "__main__":
    main()
