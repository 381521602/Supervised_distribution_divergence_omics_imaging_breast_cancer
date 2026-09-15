#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Activation-function comparison for gated NSRE-weighted FullSizeCNN."""

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
from run_multistream_cnn import nsre_scores, spiral_order  # noqa: E402
from run_nsre_weight_variants import weighted_images  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "activation_fullsize_cnn_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 50
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
SIZES = {"mRNA": 15, "CNV": 8, "miRNA": 15}
CHANNELS = 32


def mish(x):
    return x * torch.tanh(F.softplus(x))


ACTIVATIONS = {
    "ReLU": nn.ReLU(inplace=True),
    "Swish": nn.SiLU(inplace=True),
    "Mish": nn.Mish(inplace=True),
    "LeakyReLU": nn.LeakyReLU(0.1, inplace=True),
}


class GatedBranch(nn.Module):
    def __init__(self, size, channels, activation):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size=(size, size))
        self.bn = nn.BatchNorm2d(channels)
        self.gate = nn.Linear(channels, channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = self.activation(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x * torch.sigmoid(self.gate(x))


class GatedMultiStream(nn.Module):
    def __init__(self, out_dim, activation):
        super().__init__()
        self.mrna = GatedBranch(SIZES["mRNA"], CHANNELS, activation)
        self.cnv = GatedBranch(SIZES["CNV"], CHANNELS, activation)
        self.mirna = GatedBranch(SIZES["miRNA"], CHANNELS, activation)
        self.head = nn.Sequential(
            nn.Linear(CHANNELS * 3, 128), activation, nn.Dropout(0.3),
            nn.Linear(128, 64), activation, nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )
    def forward(self, x1, x2, x3):
        return self.head(torch.cat([self.mrna(x1), self.cnv(x2), self.mirna(x3)], dim=1))


def torch_seed():
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)


def run_activation(act_name):
    act = ACTIVATIONS[act_name]
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
            scores = nsre_scores(Xtr, y[tr]); order = np.argsort(scores)[::-1]
            train_imgs.append(torch.tensor(weighted_images(Xtr, SIZES[omics], order, scores), dtype=torch.float32))
            test_imgs.append(torch.tensor(weighted_images(Xte, SIZES[omics], order, scores), dtype=torch.float32))
        model = GatedMultiStream(len(enc.classes_), act); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); crit = nn.CrossEntropyLoss(); yt = torch.tensor(y[tr], dtype=torch.long)
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
    return float(np.mean(accs)), float(np.mean(f1s))


def main() -> None:
    rows = []
    for act_name in ACTIVATIONS:
        acc, f1 = run_activation(act_name)
        rows.append({"activation": act_name, "metric": "accuracy", "value": round(acc, 4)})
        rows.append({"activation": act_name, "metric": "macro_f1", "value": round(f1, 4)})
        print(f"{act_name}: acc={acc:.4f} f1={f1:.4f}")
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["activation", "metric", "value"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
