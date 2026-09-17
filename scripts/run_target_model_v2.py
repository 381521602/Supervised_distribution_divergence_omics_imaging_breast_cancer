#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target v2: JSD-weighted full-size multi-stream FullSizeCNN (ReLU, no gate)."""

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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import jsd_scores, spiral_order  # noqa: E402
from run_jsd_weight_variants import weighted_images  # noqa: E402
from run_fullsize_cnn_survival import cox_loss  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "target_model_v2_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
SIZES = {"mRNA": 15, "CNV": 8, "miRNA": 15}
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


class Multi(nn.Module):
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


class Single(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.branch = ConvBranch(size, CHANNELS)
        self.head = nn.Sequential(nn.Linear(CHANNELS, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))
    def forward(self, x):
        return self.head(self.branch(x))


def torch_seed():
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)


def make_weighted(X, size, order, scores):
    return weighted_images(X, size, order, scores)


def eval_pam50_single(omics):
    matrix = pd.read_csv(FEAT_DIR / f"{omics}_PAM50_4class_matrix.tsv", sep="\t", index_col=0)
    cases = list(matrix.index); X = matrix.loc[cases].to_numpy(dtype=float)
    lab = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str}).set_index("case_id").loc[cases]
    y_raw = lab["pam50_4class"].values; enc = LabelEncoder(); y = enc.fit_transform(y_raw)
    size = SIZES[omics]
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, te in cv.split(np.zeros(len(cases)), y):
        scaler = StandardScaler().fit(X[tr]); Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
        scores = jsd_scores(Xtr, y[tr]); order = np.argsort(scores)[::-1]
        Xtr_img = torch.tensor(make_weighted(Xtr, size, order, scores), dtype=torch.float32)
        Xte_img = torch.tensor(make_weighted(Xte, size, order, scores), dtype=torch.float32)
        model = Single(size, len(enc.classes_)); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); crit = nn.CrossEntropyLoss(); yt = torch.tensor(y[tr], dtype=torch.long)
        n = Xtr_img.shape[0]; model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                if idx.shape[0] < 2: continue
                opt.zero_grad(); loss = crit(model(Xtr_img[idx]), yt[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): pred = model(Xte_img).argmax(dim=1).numpy()
        accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.std(f1s))


def eval_pam50_multi():
    matrices = [pd.read_csv(FEAT_DIR / f"{omics}_PAM50_4class_matrix.tsv", sep="\t", index_col=0) for omics in OMICS]
    common = matrices[0].index
    for m in matrices[1:]: common = common.intersection(m.index)
    common = list(common); X_blocks = [m.loc[common].to_numpy(dtype=float) for m in matrices]
    lab = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str}).set_index("case_id").loc[common]
    y_raw = lab["pam50_4class"].values; enc = LabelEncoder(); y = enc.fit_transform(y_raw)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, te in cv.split(np.zeros(len(common)), y):
        train_imgs, test_imgs = [], []
        for block, omics in zip(X_blocks, OMICS):
            scaler = StandardScaler().fit(block[tr]); Xtr = scaler.transform(block[tr]); Xte = scaler.transform(block[te])
            scores = jsd_scores(Xtr, y[tr]); order = np.argsort(scores)[::-1]
            train_imgs.append(torch.tensor(make_weighted(Xtr, SIZES[omics], order, scores), dtype=torch.float32))
            test_imgs.append(torch.tensor(make_weighted(Xte, SIZES[omics], order, scores), dtype=torch.float32))
        model = Multi(len(enc.classes_)); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); crit = nn.CrossEntropyLoss(); yt = torch.tensor(y[tr], dtype=torch.long)
        n = train_imgs[0].shape[0]; model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                if idx.shape[0] < 2: continue
                opt.zero_grad(); loss = crit(model(train_imgs[0][idx], train_imgs[1][idx], train_imgs[2][idx]), yt[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): pred = model(test_imgs[0], test_imgs[1], test_imgs[2]).argmax(dim=1).numpy()
        accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.std(f1s))


def eval_os_single(omics):
    matrix = pd.read_csv(FEAT_DIR / f"{omics}_OS_matrix.tsv", sep="\t", index_col=0)
    cases = list(matrix.index); X = matrix.loc[cases].to_numpy(dtype=float)
    lab = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str}).set_index("case_id").loc[cases]
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
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(cis)), float(np.std(cis))


def eval_os_multi():
    matrices = [pd.read_csv(FEAT_DIR / f"{omics}_OS_matrix.tsv", sep="\t", index_col=0) for omics in OMICS]
    common = matrices[0].index
    for m in matrices[1:]: common = common.intersection(m.index)
    common = list(common); X_blocks = [m.loc[common].to_numpy(dtype=float) for m in matrices]
    lab = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str}).set_index("case_id").loc[common]
    y_event = lab["os_event"].astype(int).values; y_time = lab["os_time_days"].astype(float).values
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, cis = [], []
    for tr, te in cv.split(np.zeros(len(common)), y_event):
        train_imgs, test_imgs = [], []
        for block, omics in zip(X_blocks, OMICS):
            scaler = StandardScaler().fit(block[tr]); Xtr = scaler.transform(block[tr]); Xte = scaler.transform(block[te])
            scores = jsd_scores(Xtr, y_event[tr]); order = np.argsort(scores)[::-1]
            train_imgs.append(torch.tensor(make_weighted(Xtr, SIZES[omics], order, scores), dtype=torch.float32))
            test_imgs.append(torch.tensor(make_weighted(Xte, SIZES[omics], order, scores), dtype=torch.float32))
        model = Multi(1); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); crit = nn.BCEWithLogitsLoss(); yt = torch.tensor(y_event[tr], dtype=torch.float32).view(-1,1)
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
        model = Multi(1); torch_seed(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); time_t = torch.tensor(y_time[tr], dtype=torch.float32); event_t = torch.tensor(y_event[tr], dtype=torch.float32)
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
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(cis)), float(np.std(cis))


def main() -> None:
    rows = []
    for omics in OMICS:
        acc, acc_std, f1, f1_std = eval_pam50_single(omics)
        rows.append({"model": f"V2_{omics}", "task": "PAM50", "metric": "accuracy", "mean": round(acc,4), "std": round(acc_std,4)})
        rows.append({"model": f"V2_{omics}", "task": "PAM50", "metric": "macro_f1", "mean": round(f1,4), "std": round(f1_std,4)})
        auc, auc_std, ci, ci_std = eval_os_single(omics)
        rows.append({"model": f"V2_{omics}", "task": "OS", "metric": "roc_auc", "mean": round(auc,4), "std": round(auc_std,4)})
        rows.append({"model": f"V2_{omics}", "task": "OS", "metric": "c_index", "mean": round(ci,4), "std": round(ci_std,4)})
    acc, acc_std, f1, f1_std = eval_pam50_multi()
    rows.append({"model": "V2_Multi", "task": "PAM50", "metric": "accuracy", "mean": round(acc,4), "std": round(acc_std,4)})
    rows.append({"model": "V2_Multi", "task": "PAM50", "metric": "macro_f1", "mean": round(f1,4), "std": round(f1_std,4)})
    auc, auc_std, ci, ci_std = eval_os_multi()
    rows.append({"model": "V2_Multi", "task": "OS", "metric": "roc_auc", "mean": round(auc,4), "std": round(auc_std,4)})
    rows.append({"model": "V2_Multi", "task": "OS", "metric": "c_index", "mean": round(ci,4), "std": round(ci_std,4)})
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "task", "metric", "mean", "std"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
