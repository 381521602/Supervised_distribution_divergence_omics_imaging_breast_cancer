#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PAM50 3x-feature JSD-weighted FullSizeCNN multi-stream."""

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
from adaptive_jsd import select_features  # noqa: E402
from run_multistream_cnn import jsd_scores, spiral_order  # noqa: E402
from run_jsd_weight_variants import weighted_images  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
XENA = DATA / "external" / "xena"
OUT = DATA / "pam50_3x_fullsize_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
K3 = {"mRNA": 600, "CNV": 150, "miRNA": 600}
SIZES = {"mRNA": 25, "CNV": 13, "miRNA": 25}
CHANNELS = 32


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


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    matrices = []
    xena_paths = {"mRNA": XENA / "HiSeqV2", "CNV": XENA / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes", "miRNA": XENA / "miRNA_HiSeq_gene"}
    for omics in OMICS:
        matrix = pd.read_csv(xena_paths[omics], sep="\t", index_col=0, low_memory=False)
        _, reduced = select_features(matrix, labels, omics, "PAM50_4class", k=K3[omics])
        matrices.append(reduced)
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
        model = MultiStream(len(enc.classes_)); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); crit = nn.CrossEntropyLoss(); yt = torch.tensor(y[tr], dtype=torch.long)
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
        {"model": "PAM50_3x_JSD_Weighted_FullSizeCNN", "metric": "accuracy", "value": round(float(np.mean(accs)), 4)},
        {"model": "PAM50_3x_JSD_Weighted_FullSizeCNN", "metric": "macro_f1", "value": round(float(np.mean(f1s)), 4)},
    ]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"Accuracy={rows[0]['value']} Macro-F1={rows[1]['value']}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
