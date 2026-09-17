#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FullSizeCNN survival with 3x retained features."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from adaptive_jsd import select_features  # noqa: E402
from run_multistream_cnn import jsd_scores, spiral_order  # noqa: E402
from run_jsd_weight_variants import weighted_images  # noqa: E402
from run_fullsize_cnn_survival import cox_loss  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
XENA = DATA / "external" / "xena"
OUT = DATA / "fullsize_cnn_survival_3x_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 50
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
    def __init__(self, out_dim=1):
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


class Single(nn.Module):
    def __init__(self, size, out_dim=1):
        super().__init__()
        self.branch = ConvBranch(size, CHANNELS)
        self.head = nn.Sequential(nn.Linear(CHANNELS, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))
    def forward(self, x):
        return self.head(self.branch(x))


def torch_seed():
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)


def make_weighted(X, size, order, scores):
    return weighted_images(X, size, order, scores)


def run_single(omics, matrix, labels):
    cases = list(matrix.index); X = matrix.loc[cases].to_numpy(dtype=float)
    lab = labels.set_index("case_id").loc[cases]
    y_event = lab["os_event"].astype(int).values; y_time = lab["os_time_days"].astype(float).values
    size = SIZES[omics]
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, cis = [], []
    for tr, te in cv.split(np.zeros(len(cases)), y_event):
        scaler = StandardScaler().fit(X[tr]); Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
        scores = jsd_scores(Xtr, y_event[tr]); order = np.argsort(scores)[::-1]
        Xtr_img = torch.tensor(make_weighted(Xtr, size, order, scores), dtype=torch.float32)
        Xte_img = torch.tensor(make_weighted(Xte, size, order, scores), dtype=torch.float32)
        model = Single(size, 1); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); crit = nn.BCEWithLogitsLoss(); yt = torch.tensor(y_event[tr], dtype=torch.float32).view(-1,1)
        n = Xtr_img.shape[0]; model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                if idx.shape[0] < 2: continue
                opt.zero_grad(); loss = crit(model(Xtr_img[idx]), yt[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): prob = torch.sigmoid(model(Xte_img)).numpy().ravel()
        aucs.append(roc_auc_score(y_event[te], prob))
        model = Single(size, 1); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); time_t = torch.tensor(y_time[tr], dtype=torch.float32); event_t = torch.tensor(y_event[tr], dtype=torch.float32)
        n = Xtr_img.shape[0]; model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                if idx.shape[0] < 2: continue
                opt.zero_grad(); loss = cox_loss(model(Xtr_img[idx]), time_t[idx], event_t[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): hazard = model(Xte_img).numpy().ravel()
        cis.append(concordance_index(y_time[te], -hazard, y_event[te]))
    return float(np.mean(aucs)), float(np.mean(cis))


def run_multi(matrices, labels):
    common = matrices[0].index
    for m in matrices[1:]: common = common.intersection(m.index)
    common = list(common); blocks = [m.loc[common].to_numpy(dtype=float) for m in matrices]
    lab = labels.set_index("case_id").loc[common]; y_event = lab["os_event"].astype(int).values; y_time = lab["os_time_days"].astype(float).values
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, cis = [], []
    for tr, te in cv.split(np.zeros(len(common)), y_event):
        train_imgs, test_imgs = [], []
        for block, omics in zip(blocks, OMICS):
            scaler = StandardScaler().fit(block[tr]); Xtr = scaler.transform(block[tr]); Xte = scaler.transform(block[te])
            scores = jsd_scores(Xtr, y_event[tr]); order = np.argsort(scores)[::-1]
            train_imgs.append(torch.tensor(make_weighted(Xtr, SIZES[omics], order, scores), dtype=torch.float32))
            test_imgs.append(torch.tensor(make_weighted(Xte, SIZES[omics], order, scores), dtype=torch.float32))
        model = MultiStream(1); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); crit = nn.BCEWithLogitsLoss(); yt = torch.tensor(y_event[tr], dtype=torch.float32).view(-1,1)
        n = train_imgs[0].shape[0]; model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                if idx.shape[0] < 2: continue
                opt.zero_grad(); loss = crit(model(train_imgs[0][idx], train_imgs[1][idx], train_imgs[2][idx]), yt[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): prob = torch.sigmoid(model(test_imgs[0], test_imgs[1], test_imgs[2])).numpy().ravel()
        aucs.append(roc_auc_score(y_event[te], prob))
        model = MultiStream(1); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); time_t = torch.tensor(y_time[tr], dtype=torch.float32); event_t = torch.tensor(y_event[tr], dtype=torch.float32)
        n = train_imgs[0].shape[0]; model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                if idx.shape[0] < 2: continue
                opt.zero_grad(); loss = cox_loss(model(train_imgs[0][idx], train_imgs[1][idx], train_imgs[2][idx]), time_t[idx], event_t[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): hazard = model(test_imgs[0], test_imgs[1], test_imgs[2]).numpy().ravel()
        cis.append(concordance_index(y_time[te], -hazard, y_event[te]))
    return float(np.mean(aucs)), float(np.mean(cis))


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    rows = []
    matrices = {}
    xena_paths = {"mRNA": XENA / "HiSeqV2", "CNV": XENA / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes", "miRNA": XENA / "miRNA_HiSeq_gene"}
    for omics in OMICS:
        matrix = pd.read_csv(xena_paths[omics], sep="\t", index_col=0, low_memory=False)
        _, reduced = select_features(matrix, labels, omics, "OS", k=K3[omics])
        matrices[omics] = reduced
        auc, ci = run_single(omics, reduced, labels)
        rows.append({"model": f"3x_{omics}", "metric": "roc_auc", "value": round(auc, 4)})
        rows.append({"model": f"3x_{omics}", "metric": "c_index", "value": round(ci, 4)})
        print(f"{omics}: AUC={auc:.4f} C-index={ci:.4f}")
    auc, ci = run_multi([matrices[o] for o in OMICS], labels)
    rows.append({"model": "3x_MultiStream", "metric": "roc_auc", "value": round(auc, 4)})
    rows.append({"model": "3x_MultiStream", "metric": "c_index", "value": round(ci, 4)})
    print(f"MultiStream: AUC={auc:.4f} C-index={ci:.4f}")
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
