#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mRNA(600) + CNV(50) two-stream FullSizeCNN for survival."""

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
OUT = DATA / "mrna600_cnv50_survival_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
K = {"mRNA": 600, "CNV": 50}
SIZES = {"mRNA": 25, "CNV": 8}
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


class TwoStream(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.mrna = ConvBranch(SIZES["mRNA"], CHANNELS)
        self.cnv = ConvBranch(SIZES["CNV"], CHANNELS)
        self.head = nn.Sequential(
            nn.Linear(CHANNELS * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )
    def forward(self, x_mrna, x_cnv):
        return self.head(torch.cat([self.mrna(x_mrna), self.cnv(x_cnv)], dim=1))


def torch_seed():
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)


def make_weighted(X, size, order, scores):
    return weighted_images(X, size, order, scores)


def run_fusion(mrna_mat, cnv_mat, labels):
    common = mrna_mat.index.intersection(cnv_mat.index)
    common = list(common)
    blocks = [mrna_mat.loc[common].to_numpy(dtype=float), cnv_mat.loc[common].to_numpy(dtype=float)]
    lab = labels.set_index("case_id").loc[common]
    y_event = lab["os_event"].astype(int).values
    y_time = lab["os_time_days"].astype(float).values
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, cis = [], []
    for tr, te in cv.split(np.zeros(len(common)), y_event):
        train_imgs, test_imgs = [], []
        for block, omics in zip(blocks, ["mRNA", "CNV"]):
            scaler = StandardScaler().fit(block[tr]); Xtr = scaler.transform(block[tr]); Xte = scaler.transform(block[te])
            scores = jsd_scores(Xtr, y_event[tr]); order = np.argsort(scores)[::-1]
            train_imgs.append(torch.tensor(make_weighted(Xtr, SIZES[omics], order, scores), dtype=torch.float32))
            test_imgs.append(torch.tensor(make_weighted(Xte, SIZES[omics], order, scores), dtype=torch.float32))
        model = TwoStream(1); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); crit = nn.BCEWithLogitsLoss(); yt = torch.tensor(y_event[tr], dtype=torch.float32).view(-1,1)
        n = train_imgs[0].shape[0]; model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                if idx.shape[0] < 2: continue
                opt.zero_grad(); loss = crit(model(train_imgs[0][idx], train_imgs[1][idx]), yt[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): prob = torch.sigmoid(model(test_imgs[0], test_imgs[1])).numpy().ravel()
        aucs.append(roc_auc_score(y_event[te], prob))
        model = TwoStream(1); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); time_t = torch.tensor(y_time[tr], dtype=torch.float32); event_t = torch.tensor(y_event[tr], dtype=torch.float32)
        n = train_imgs[0].shape[0]; model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                if idx.shape[0] < 2: continue
                opt.zero_grad(); loss = cox_loss(model(train_imgs[0][idx], train_imgs[1][idx]), time_t[idx], event_t[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): hazard = model(test_imgs[0], test_imgs[1]).numpy().ravel()
        cis.append(concordance_index(y_time[te], -hazard, y_event[te]))
    return float(np.mean(aucs)), float(np.mean(cis))


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    xena_paths = {"mRNA": XENA / "HiSeqV2", "CNV": XENA / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes"}
    mrna = pd.read_csv(xena_paths["mRNA"], sep="\t", index_col=0, low_memory=False)
    cnv = pd.read_csv(xena_paths["CNV"], sep="\t", index_col=0, low_memory=False)
    _, mrna_red = select_features(mrna, labels, "mRNA", "OS", k=K["mRNA"])
    _, cnv_red = select_features(cnv, labels, "CNV", "OS", k=K["CNV"])
    auc, ci = run_fusion(mrna_red, cnv_red, labels)
    rows = [
        {"model": "mRNA600_CNV50_FullSizeCNN", "metric": "roc_auc", "value": round(auc, 4)},
        {"model": "mRNA600_CNV50_FullSizeCNN", "metric": "c_index", "value": round(ci, 4)},
    ]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"AUC={auc:.4f} C-index={ci:.4f}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
