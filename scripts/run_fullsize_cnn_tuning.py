#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small hyperparameter search for full-size multi-stream CNN on PAM50."""

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
OUT = DATA / "fullsize_cnn_tuning_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 80
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
SIZES = {"mRNA": 15, "CNV": 8, "miRNA": 15}


class ConvBranch(nn.Module):
    def __init__(self, size, channels):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size=(size, size))
        self.bn = nn.BatchNorm2d(channels)
    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.flatten(1)


class MultiStream(nn.Module):
    def __init__(self, out_dim, channels, dropout):
        super().__init__()
        self.mrna = ConvBranch(SIZES["mRNA"], channels)
        self.cnv = ConvBranch(SIZES["CNV"], channels)
        self.mirna = ConvBranch(SIZES["miRNA"], channels)
        self.head = nn.Sequential(
            nn.Linear(channels * 3, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )
    def forward(self, x1, x2, x3):
        return self.head(torch.cat([self.mrna(x1), self.cnv(x2), self.mirna(x3)], dim=1))


def torch_seed():
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)


def run_fold(train_imgs, val_imgs, test_imgs, y_train, y_val, y_test, out_dim, channels, dropout):
    torch_seed()
    model = MultiStream(out_dim, channels, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    X_train = [torch.tensor(x, dtype=torch.float32) for x in train_imgs]
    X_val = [torch.tensor(x, dtype=torch.float32) for x in val_imgs]
    X_test = [torch.tensor(x, dtype=torch.float32) for x in test_imgs]
    yt = torch.tensor(y_train, dtype=torch.long); yv = torch.tensor(y_val, dtype=torch.long)
    n = X_train[0].shape[0]
    best_val = float("inf"); best_state = None; bad = 0
    for _ in range(EPOCHS):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            if idx.shape[0] < 2: continue
            opt.zero_grad(); loss = crit(model(X_train[0][idx], X_train[1][idx], X_train[2][idx]), yt[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_val[0], X_val[1], X_val[2]), yv).item()
        if val_loss < best_val - 1e-4:
            best_val = val_loss; best_state = model.state_dict().copy(); bad = 0
        else:
            bad += 1
            if bad >= 10: break
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
    for m in matrices[1:]: common = common.intersection(m.index)
    common = list(common)
    X_blocks = [m.loc[common].to_numpy(dtype=float) for m in matrices]
    y_raw = labels.set_index("case_id").loc[common, "pam50_4class"].values
    enc = LabelEncoder(); y = enc.fit_transform(y_raw)
    rows = []
    for channels in [16, 32, 64]:
        for dropout in [0.3, 0.5]:
            cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            accs, f1s = [], []
            for tr, te in cv.split(np.zeros(len(common)), y):
                tr, va = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=RANDOM_STATE)
                def make(idx):
                    imgs = []
                    for block, omics in zip(X_blocks, OMICS):
                        scaler = StandardScaler().fit(block[tr]); Xsub = scaler.transform(block[idx])
                        order = feature_order(scaler.transform(block[tr]), y[tr])
                        imgs.append(to_images_with_order(Xsub, SIZES[omics], order))
                    return imgs
                pred = run_fold(make(tr), make(va), make(te), y[tr], y[va], y[te], len(enc.classes_), channels, dropout)
                accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
            acc = float(np.mean(accs)); f1 = float(np.mean(f1s))
            rows.append({"channels": channels, "dropout": dropout, "metric": "accuracy", "value": round(acc, 4)})
            rows.append({"channels": channels, "dropout": dropout, "metric": "macro_f1", "value": round(f1, 4)})
            print(f"channels={channels} dropout={dropout}: acc={acc:.4f} f1={f1:.4f}")
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["channels", "dropout", "metric", "value"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
