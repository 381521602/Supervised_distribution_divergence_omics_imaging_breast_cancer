#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FullSizeCNN with Inception modules on JSD-weighted images."""

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
from run_multistream_cnn import jsd_scores, spiral_order  # noqa: E402
from run_jsd_weight_variants import weighted_images  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "inception_fullsize_cnn_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
SIZES = {"mRNA": 15, "CNV": 8, "miRNA": 15}
CHANNELS = 32


class InceptionBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b1 = nn.Conv2d(1, channels, 1)
        self.b2 = nn.Conv2d(1, channels, 3, padding=1)
        self.b3 = nn.Conv2d(1, channels, 5, padding=2)
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.b4 = nn.Conv2d(1, channels, 1)
        self.bn = nn.BatchNorm2d(channels * 4)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(self.pool(x))], dim=1)
        out = self.bn(out)
        out = F.relu(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return out.flatten(1)


class InceptionMultiStream(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mrna = InceptionBranch(CHANNELS)
        self.cnv = InceptionBranch(CHANNELS)
        self.mirna = InceptionBranch(CHANNELS)
        self.head = nn.Sequential(
            nn.Linear(CHANNELS * 4 * 3, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )
    def forward(self, x1, x2, x3):
        return self.head(torch.cat([self.mrna(x1), self.cnv(x2), self.mirna(x3)], dim=1))


def torch_seed():
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)


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
            scaler = StandardScaler().fit(block[tr]); Xtr = scaler.transform(block[tr]); Xte = scaler.transform(block[te])
            scores = jsd_scores(Xtr, y[tr]); order = np.argsort(scores)[::-1]
            train_imgs.append(torch.tensor(weighted_images(Xtr, SIZES[omics], order, scores), dtype=torch.float32))
            test_imgs.append(torch.tensor(weighted_images(Xte, SIZES[omics], order, scores), dtype=torch.float32))
        model = InceptionMultiStream(len(enc.classes_)); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); crit = nn.CrossEntropyLoss(); yt = torch.tensor(y[tr], dtype=torch.long)
        n = train_imgs[0].shape[0]; model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                if idx.shape[0] < 2: continue
                opt.zero_grad(); loss = crit(model(train_imgs[0][idx], train_imgs[1][idx], train_imgs[2][idx]), yt[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(test_imgs[0], test_imgs[1], test_imgs[2])
        pred = logits.argmax(dim=1).numpy()
        accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
    rows = [
        {"model": "Inception_JSD_Weighted_FullSizeCNN", "metric": "accuracy", "value": round(float(np.mean(accs)), 4)},
        {"model": "Inception_JSD_Weighted_FullSizeCNN", "metric": "macro_f1", "value": round(float(np.mean(f1s)), 4)},
    ]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"Accuracy={rows[0]['value']} Macro-F1={rows[1]['value']}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
