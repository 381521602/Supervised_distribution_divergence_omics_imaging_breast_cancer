#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FullSizeCNN survival prediction: binary AUC and Cox C-index."""

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
from run_multistream_cnn import feature_order, to_images_with_order  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "fullsize_cnn_survival_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
SIZES = {"mRNA": 15, "CNV": 8, "miRNA": 15}
CHANNELS = 32


class FullSizeCNN(nn.Module):
    def __init__(self, size, out_dim=1, channels=CHANNELS):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size=(size, size))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.head(self.conv(x))


class MultiStreamFullSize(nn.Module):
    def __init__(self, out_dim=1, channels=CHANNELS):
        super().__init__()
        self.mrna = nn.Conv2d(1, channels, kernel_size=(15, 15))
        self.cnv = nn.Conv2d(1, channels, kernel_size=(8, 8))
        self.mirna = nn.Conv2d(1, channels, kernel_size=(15, 15))
        self.head = nn.Sequential(
            nn.Linear(channels * 3, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x_mrna, x_cnv, x_mirna):
        f1 = self.mrna(x_mrna).flatten(1)
        f2 = self.cnv(x_cnv).flatten(1)
        f3 = self.mirna(x_mirna).flatten(1)
        return self.head(torch.cat([f1, f2, f3], dim=1))


def torch_seed():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def cox_loss(log_h, time, event):
    risk = log_h.view(-1)
    _, idx = torch.sort(time, descending=True)
    risk = risk[idx]
    event = event[idx]
    exp_risk = torch.exp(risk)
    cumsum = torch.cumsum(exp_risk, dim=0)
    return -torch.mean((risk - torch.log(cumsum + 1e-8)) * event.float())


def train_binary(model, X_train, y_train):
    torch_seed()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    n = X_train.shape[0]
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad(); loss = crit(model(X_train[idx]), yt[idx]); loss.backward(); opt.step()


def train_cox(model, X_train, y_time, y_event):
    torch_seed()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    time_t = torch.tensor(y_time, dtype=torch.float32)
    event_t = torch.tensor(y_event, dtype=torch.float32)
    n = X_train.shape[0]
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad(); loss = cox_loss(model(X_train[idx]), time_t[idx], event_t[idx]); loss.backward(); opt.step()


def run_single(omics, matrix, labels):
    cases = list(matrix.index)
    X = matrix.loc[cases].to_numpy(dtype=float)
    lab = labels.set_index("case_id").loc[cases]
    y_event = lab["os_event"].astype(int).values
    y_time = lab["os_time_days"].astype(float).values
    size = SIZES[omics]
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, cis = [], []
    for tr, te in cv.split(np.zeros(len(cases)), y_event):
        scaler = StandardScaler().fit(X[tr]); Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
        order = feature_order(Xtr, y_event[tr])
        Xtr_img = torch.tensor(to_images_with_order(Xtr, size, order), dtype=torch.float32)
        Xte_img = torch.tensor(to_images_with_order(Xte, size, order), dtype=torch.float32)
        model = FullSizeCNN(size, 1)
        train_binary(model, Xtr_img, y_event[tr])
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(Xte_img)).numpy().ravel()
        aucs.append(roc_auc_score(y_event[te], prob))
        model = FullSizeCNN(size, 1)
        train_cox(model, Xtr_img, y_time[tr], y_event[tr])
        model.eval()
        with torch.no_grad():
            hazard = model(Xte_img).numpy().ravel()
        cis.append(concordance_index(y_time[te], -hazard, y_event[te]))
    return float(np.mean(aucs)), float(np.mean(cis))


def run_multi(matrices, labels):
    common = matrices[0].index
    for m in matrices[1:]:
        common = common.intersection(m.index)
    common = list(common)
    blocks = [m.loc[common].to_numpy(dtype=float) for m in matrices]
    lab = labels.set_index("case_id").loc[common]
    y_event = lab["os_event"].astype(int).values
    y_time = lab["os_time_days"].astype(float).values
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, cis = [], []
    for tr, te in cv.split(np.zeros(len(common)), y_event):
        train_imgs, test_imgs = [], []
        for block, omics in zip(blocks, OMICS):
            scaler = StandardScaler().fit(block[tr]); Xtr = scaler.transform(block[tr]); Xte = scaler.transform(block[te])
            order = feature_order(Xtr, y_event[tr])
            train_imgs.append(torch.tensor(to_images_with_order(Xtr, SIZES[omics], order), dtype=torch.float32))
            test_imgs.append(torch.tensor(to_images_with_order(Xte, SIZES[omics], order), dtype=torch.float32))
        model = MultiStreamFullSize(1)
        torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); crit = nn.BCEWithLogitsLoss(); yt = torch.tensor(y_event[tr], dtype=torch.float32).view(-1,1)
        n = train_imgs[0].shape[0]; model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                if idx.shape[0] < 2: continue
                opt.zero_grad(); loss = crit(model(train_imgs[0][idx], train_imgs[1][idx], train_imgs[2][idx]), yt[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(test_imgs[0], test_imgs[1], test_imgs[2])).numpy().ravel()
        aucs.append(roc_auc_score(y_event[te], prob))
        model = MultiStreamFullSize(1)
        torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        time_t = torch.tensor(y_time[tr], dtype=torch.float32); event_t = torch.tensor(y_event[tr], dtype=torch.float32)
        n = train_imgs[0].shape[0]; model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                if idx.shape[0] < 2: continue
                opt.zero_grad(); loss = cox_loss(model(train_imgs[0][idx], train_imgs[1][idx], train_imgs[2][idx]), time_t[idx], event_t[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            hazard = model(test_imgs[0], test_imgs[1], test_imgs[2]).numpy().ravel()
        cis.append(concordance_index(y_time[te], -hazard, y_event[te]))
    return float(np.mean(aucs)), float(np.mean(cis))


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    rows = []
    matrices = {}
    for omics in OMICS:
        matrix = pd.read_csv(FEAT_DIR / f"{omics}_OS_matrix.tsv", sep="\t", index_col=0)
        matrices[omics] = matrix
        auc, ci = run_single(omics, matrix, labels)
        rows.append({"model": f"FullSizeCNN_{omics}", "metric": "roc_auc", "value": round(auc, 4)})
        rows.append({"model": f"FullSizeCNN_{omics}", "metric": "c_index", "value": round(ci, 4)})
        print(f"{omics}: AUC={auc:.4f} C-index={ci:.4f}")
    auc, ci = run_multi([matrices[o] for o in OMICS], labels)
    rows.append({"model": "FullSizeCNN_MultiStream", "metric": "roc_auc", "value": round(auc, 4)})
    rows.append({"model": "FullSizeCNN_MultiStream", "metric": "c_index", "value": round(ci, 4)})
    print(f"MultiStream: AUC={auc:.4f} C-index={ci:.4f}")
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
