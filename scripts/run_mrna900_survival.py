#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mRNA 900-feature JSD-weighted FullSizeCNN for survival."""

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
OUT = DATA / "mrna900_survival_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
K = 900
SIZE = 30
CHANNELS = 32


class FullSizeCNN(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.conv = nn.Conv2d(1, CHANNELS, kernel_size=(SIZE, SIZE))
        self.bn = nn.BatchNorm2d(CHANNELS)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(CHANNELS, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim)
        )
    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return self.head(x.flatten(1))


def torch_seed():
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    mrna = pd.read_csv(XENA / "HiSeqV2", sep="\t", index_col=0, low_memory=False)
    _, reduced = select_features(mrna, labels, "mRNA", "OS", k=K)
    cases = list(reduced.index)
    X = reduced.loc[cases].to_numpy(dtype=float)
    lab = labels.set_index("case_id").loc[cases]
    y_event = lab["os_event"].astype(int).values
    y_time = lab["os_time_days"].astype(float).values
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, cis = [], []
    for tr, te in cv.split(np.zeros(len(cases)), y_event):
        scaler = StandardScaler().fit(X[tr]); Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
        scores = jsd_scores(Xtr, y_event[tr]); order = np.argsort(scores)[::-1]
        Xtr_img = torch.tensor(weighted_images(Xtr, SIZE, order, scores), dtype=torch.float32)
        Xte_img = torch.tensor(weighted_images(Xte, SIZE, order, scores), dtype=torch.float32)
        model = FullSizeCNN(1); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); crit = nn.BCEWithLogitsLoss(); yt = torch.tensor(y_event[tr], dtype=torch.float32).view(-1,1)
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
        model = FullSizeCNN(1); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); time_t = torch.tensor(y_time[tr], dtype=torch.float32); event_t = torch.tensor(y_event[tr], dtype=torch.float32)
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
    rows = [
        {"model": "mRNA900_JSD_Weighted_FullSizeCNN", "metric": "roc_auc", "value": round(float(np.mean(aucs)), 4)},
        {"model": "mRNA900_JSD_Weighted_FullSizeCNN", "metric": "c_index", "value": round(float(np.mean(cis)), 4)},
    ]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"AUC={rows[0]['value']} C-index={rows[1]['value']}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
