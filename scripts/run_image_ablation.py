#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordering and image-size ablation for the improved single-omics CNN."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import feature_order, to_images_with_order, spiral_order  # noqa: E402
from run_single_omics_cnn_v2 import SmallResCNN  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "image_ablation_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
SIZES = {"mRNA": [15, 17], "CNV": [8, 10], "miRNA": [15, 17]}


def mean_order(X: np.ndarray) -> np.ndarray:
    means = np.mean(X, axis=0)
    return np.argsort(means)[::-1]


def random_order(X: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.permutation(X.shape[1])


def torch_seed():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def run_fold(X_train, X_test, y_train, y_test, out_dim):
    torch_seed()
    model = SmallResCNN(out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
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


def evaluate(omics: str, ordering: str, size: int, X, y):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for train_idx, test_idx in cv.split(np.zeros(len(y)), y):
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        if ordering == "JSD":
            order = feature_order(X_train, y[train_idx])
        elif ordering == "mean":
            order = mean_order(X_train)
        else:
            order = random_order(X_train, RANDOM_STATE)
        X_train_img = to_images_with_order(X_train, size, order)
        X_test_img = to_images_with_order(X_test, size, order)
        pred = run_fold(X_train_img, X_test_img, y[train_idx], y[test_idx], len(np.unique(y)))
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s))


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

        for ordering in ["JSD", "mean", "random"]:
            for size in SIZES[omics]:
                acc, f1 = evaluate(omics, ordering, size, X, y)
                rows.append({"omics": omics, "ordering": ordering, "size": size, "metric": "accuracy", "value": round(acc, 4)})
                rows.append({"omics": omics, "ordering": ordering, "size": size, "metric": "macro_f1", "value": round(f1, 4)})
                print(f"{omics} {ordering} {size}x{size}: acc={acc:.4f} f1={f1:.4f}")

    if rows:
        columns = ["omics", "ordering", "size", "metric", "value"]
        with OUT.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
