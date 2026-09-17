#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSD 原始图像上对比五类轻量经典 CNN 与 FullSizeCNN。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parent.parent
PAM_DIR = ROOT / "data/final_datasets/PAM50"
SUR_DIR = ROOT / "data/final_datasets/Survival"
IMG = ROOT / "data/images"
OUT = ROOT / "data/jsd_classic_cnn_variants_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64


class FullSizeCNN(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.head(F.relu(self.conv(x)))


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_channels
        for _ in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(ch, growth, 3, padding=1, bias=False),
                    nn.BatchNorm2d(growth),
                    nn.ReLU(inplace=True),
                )
            )
            ch += growth

    def forward(self, x):
        for layer in self.layers:
            y = layer(x)
            x = torch.cat([x, y], dim=1)
        return x


class DenseNetLite(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DenseBlock(32, 16, 2),
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(32, out_dim))

    def forward(self, x):
        return self.head(self.features(x))


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, expand, stride=1):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        y = self.layers(x)
        return y + x if self.use_res else y


class MobileNetV2Lite(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            InvertedResidual(32, 32, 2, 1),
            InvertedResidual(32, 64, 2, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, x):
        return self.head(self.features(x))


def channel_shuffle(x, groups):
    b, c, h, w = x.shape
    if c % groups != 0:
        return x
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class ShuffleUnit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        branch_hidden = out_ch // 2
        self.branch = nn.Sequential(
            nn.Conv2d(in_ch // 2, branch_hidden, 1, bias=False),
            nn.BatchNorm2d(branch_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_hidden, branch_hidden, 3, groups=branch_hidden, padding=1, bias=False),
            nn.BatchNorm2d(branch_hidden),
            nn.Conv2d(branch_hidden, out_ch - in_ch // 2, 1, bias=False),
            nn.BatchNorm2d(out_ch - in_ch // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        out = torch.cat([x1, self.branch(x2)], dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2Lite(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ShuffleUnit(32, 32),
            ShuffleUnit(32, 64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, x):
        return self.head(self.features(x))


class InceptionBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.b1 = nn.Conv2d(in_ch, 16, 1)
        self.b2 = nn.Sequential(nn.Conv2d(in_ch, 16, 1), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1))
        self.b3 = nn.Sequential(nn.Conv2d(in_ch, 16, 1), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 5, padding=2))
        self.b4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), nn.Conv2d(in_ch, 16, 1))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class LightInception(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            InceptionBlock(32),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, x):
        return self.head(self.features(x))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = x.mean(dim=(2, 3), keepdim=True).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand, stride=1):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            SEBlock(hidden),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        y = self.layers(x)
        return y + x if self.use_res else y


class EfficientNetB0Lite(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            MBConvBlock(32, 32, 1, 1),
            MBConvBlock(32, 64, 4, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, size, out_dim):
    if name == "FullSizeCNN":
        return FullSizeCNN(size, out_dim)
    return {
        "DenseNetLite": DenseNetLite(out_dim),
        "MobileNetV2Lite": MobileNetV2Lite(out_dim),
        "ShuffleNetV2Lite": ShuffleNetV2Lite(out_dim),
        "LightInception": LightInception(out_dim),
        "EfficientNetB0Lite": EfficientNetB0Lite(out_dim),
    }[name]


def train_model(name, size, Xtr, ytr, Xte, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = build_model(name, size, 1 if binary else 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    if binary:
        crit = nn.BCEWithLogitsLoss()
        yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    else:
        crit = nn.CrossEntropyLoss()
        yt = torch.tensor(ytr, dtype=torch.long)
    n = Xt.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte, dtype=torch.float32))
    return torch.sigmoid(out).numpy().ravel() if binary else out.argmax(dim=1).numpy()


NAMES = ["FullSizeCNN", "DenseNetLite", "MobileNetV2Lite", "ShuffleNetV2Lite", "LightInception", "EfficientNetB0Lite"]


def run_pam50():
    df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
    enc = LabelEncoder()
    y = enc.fit_transform(df["pam50"].values)
    imgs = np.load(IMG / "PAM50/mRNA/images.npy")
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {name: {"acc": [], "f1": []} for name in NAMES}
    for tr, te in cv.split(np.zeros(len(y)), y):
        for name in NAMES:
            pred = train_model(name, 20, imgs[tr], y[tr], imgs[te], binary=False)
            results[name]["acc"].append(accuracy_score(y[te], pred))
            results[name]["f1"].append(f1_score(y[te], pred, average="macro"))
    rows = []
    for name, m in results.items():
        rows.append({"variant": name, "task": "PAM50", "metric": "accuracy", "mean": round(float(np.mean(m["acc"])), 4), "std": round(float(np.std(m["acc"])), 4)})
        rows.append({"variant": name, "task": "PAM50", "metric": "macro_f1", "mean": round(float(np.mean(m["f1"])), 4), "std": round(float(np.std(m["f1"])), 4)})
    return rows


def run_survival():
    df = pd.read_csv(SUR_DIR / "mRNA_Survival_final.tsv", sep="\t")
    y_event = df["os_event"].astype(int).values
    y_time = df["os_time_days"].astype(float).values
    imgs = np.load(IMG / "Survival/mRNA/images.npy")
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {name: {"auc": [], "ci": []} for name in NAMES}
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        for name in NAMES:
            prob = train_model(name, 15, imgs[tr], y_event[tr], imgs[te], binary=True)
            results[name]["auc"].append(roc_auc_score(y_event[te], prob))
            results[name]["ci"].append(concordance_index(y_time[te], -prob, y_event[te]))
    rows = []
    for name, m in results.items():
        rows.append({"variant": name, "task": "Survival", "metric": "roc_auc", "mean": round(float(np.mean(m["auc"])), 4), "std": round(float(np.std(m["auc"])), 4)})
        rows.append({"variant": name, "task": "Survival", "metric": "c_index", "mean": round(float(np.mean(m["ci"])), 4), "std": round(float(np.std(m["ci"])), 4)})
    return rows


def main():
    rows = run_pam50() + run_survival()
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}")
    for r in rows:
        print(r["variant"], r["task"], r["metric"], r["mean"], "+-", r["std"])


if __name__ == "__main__":
    main()
