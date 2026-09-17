#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gradient-based saliency + JSD key-gene mining for the target model."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import jsd_scores, spiral_order  # noqa: E402
from run_jsd_weight_variants import weighted_images  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT_DIR = DATA / "key_genes"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
OMICS = ["mRNA", "CNV", "miRNA"]
SIZES = {"mRNA": 15, "CNV": 8, "miRNA": 15}
CHANNELS = 32
TOP_N = 50


class GatedBranch(nn.Module):
    def __init__(self, size, channels):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size=(size, size))
        self.bn = nn.BatchNorm2d(channels)
        self.gate = nn.Linear(channels, channels)
    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x * torch.sigmoid(self.gate(x))


class Multi(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mrna = GatedBranch(SIZES["mRNA"], CHANNELS)
        self.cnv = GatedBranch(SIZES["CNV"], CHANNELS)
        self.mirna = GatedBranch(SIZES["miRNA"], CHANNELS)
        self.head = nn.Sequential(
            nn.Linear(CHANNELS * 3, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )
    def forward(self, x1, x2, x3):
        return self.head(torch.cat([self.mrna(x1), self.cnv(x2), self.mirna(x3)], dim=1))


def torch_seed():
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)


def make_weighted(X, size, order, scores):
    return weighted_images(X, size, order, scores)


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    matrices = [pd.read_csv(FEAT_DIR / f"{o}_PAM50_4class_matrix.tsv", sep="\t", index_col=0) for o in OMICS]
    common = matrices[0].index
    for m in matrices[1:]:
        common = common.intersection(m.index)
    common = list(common)
    X_blocks = [m.loc[common].to_numpy(dtype=float) for m in matrices]
    y_raw = labels.set_index("case_id").loc[common, "pam50_4class"].values
    enc = LabelEncoder(); y = enc.fit_transform(y_raw)

    # Fit one fold model for saliency.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tr, te = next(iter(cv.split(np.zeros(len(common)), y)))
    train_imgs, test_imgs, orders, scores_list = [], [], [], []
    for block, omics in zip(X_blocks, OMICS):
        scaler = StandardScaler().fit(block[tr]); Xtr = scaler.transform(block[tr]); Xte = scaler.transform(block[te])
        scores = jsd_scores(Xtr, y[tr]); order = np.argsort(scores)[::-1]
        orders.append(order); scores_list.append(scores)
        train_imgs.append(torch.tensor(make_weighted(Xtr, SIZES[omics], order, scores), dtype=torch.float32))
        test_imgs.append(torch.tensor(make_weighted(Xte, SIZES[omics], order, scores), dtype=torch.float32))
    model = Multi(len(enc.classes_)); torch_seed()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); crit = nn.CrossEntropyLoss(); yt = torch.tensor(y[tr], dtype=torch.long)
    n = train_imgs[0].shape[0]
    for _ in range(60):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i+64]
            if idx.shape[0] < 2: continue
            opt.zero_grad(); loss = crit(model(train_imgs[0][idx], train_imgs[1][idx], train_imgs[2][idx]), yt[idx]); loss.backward(); opt.step()
    model.eval()

    # Aggregate input-gradient saliency.
    sal_maps = {o: np.zeros((SIZES[o], SIZES[o]), dtype=float) for o in OMICS}
    for i in range(len(te)):
        imgs = [test_imgs[0][i:i+1].clone().requires_grad_(True), test_imgs[1][i:i+1].clone().requires_grad_(True), test_imgs[2][i:i+1].clone().requires_grad_(True)]
        logits = model(imgs[0], imgs[1], imgs[2])
        logits[0, y[te][i]].backward()
        for o, img in zip(OMICS, imgs):
            sal_maps[o] += np.abs(img.grad[0, 0].detach().numpy())

    rows = []
    for o in OMICS:
        positions = spiral_order(SIZES[o])
        sal = sal_maps[o]
        # rank pixels by saliency; map back to genes.
        pixel_scores = []
        oi = OMICS.index(o)
        for pos, feat_idx in zip(positions, orders[oi]):
            if feat_idx < X_blocks[OMICS.index(o)].shape[1]:
                pixel_scores.append((sal[pos], feat_idx, matrices[oi].columns[feat_idx]))
        pixel_scores.sort(reverse=True, key=lambda x: x[0])
        for rank, (salv, feat_idx, gene) in enumerate(pixel_scores[:TOP_N], 1):
            rows.append({"omics": o, "rank": rank, "gene": gene, "saliency": float(salv), "jsd": float(scores_list[oi][feat_idx])})

    with (OUT_DIR / "key_genes_saliency.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["omics", "rank", "gene", "saliency", "jsd"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} key-gene rows -> {OUT_DIR / 'key_genes_saliency.tsv'}")


if __name__ == "__main__":
    main()
