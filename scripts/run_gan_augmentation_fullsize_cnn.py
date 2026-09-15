#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""严格折内 GAN 扩增训练 FullSizeCNN。

关键：每折只在训练折上训练 GAN；测试折不参与 GAN 训练，也不参与任何标准化或扩增。
"""

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
OUT = ROOT / "data/gan_augmentation_fullsize_cnn_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
NOISE_DIM = 64
EMBED_DIM = 16
GAN_ITERS = 120
AUG_PER_CLASS = 120


class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dim, num_classes, output_size):
        super().__init__()
        self.output_size = output_size
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Sequential(nn.Linear(noise_dim + embed_dim, 64 * 4 * 4), nn.BatchNorm1d(64 * 4 * 4), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.up2 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.out = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, z, labels):
        x = self.fc(torch.cat([z, self.embed(labels)], dim=1))
        x = x.view(-1, 64, 4, 4)
        x = F.relu(self.bn1(self.up1(x)), inplace=True)
        x = F.relu(self.bn2(self.up2(x)), inplace=True)
        x = torch.tanh(self.out(x))
        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return x


class Discriminator(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Sequential(nn.Linear(64 + embed_dim, 1), nn.Sigmoid())

    def forward(self, x, labels):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.pool(x).flatten(1)
        return self.fc(torch.cat([x, self.embed(labels)], dim=1))


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


def train_fullsize(Xtr, ytr, Xte, size, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeCNN(size, 1 if binary else 4)
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


def train_gan(images, labels, num_classes, output_size):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    xmin = float(np.min(images))
    xmax = float(np.max(images))
    x_norm = ((images - xmin) / max(xmax - xmin, 1e-6)) * 2.0 - 1.0
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    G = Generator(NOISE_DIM, EMBED_DIM, num_classes, output_size)
    D = Discriminator(EMBED_DIM, num_classes)
    g_opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    n = x_tensor.shape[0]

    for _ in range(GAN_ITERS):
        idx = torch.randint(0, n, (BATCH_SIZE,))
        real = x_tensor[idx]
        real_y = y_tensor[idx]
        ones = torch.ones((real.shape[0], 1), dtype=torch.float32)
        zeros = torch.zeros((real.shape[0], 1), dtype=torch.float32)

        d_opt.zero_grad()
        real_loss = bce(D(real, real_y), ones)
        z = torch.randn(real.shape[0], NOISE_DIM)
        fake_y = torch.randint(0, num_classes, (real.shape[0],))
        fake = G(z, fake_y).detach()
        fake_loss = bce(D(fake, fake_y), zeros)
        d_loss = (real_loss + fake_loss) / 2.0
        d_loss.backward()
        d_opt.step()

        g_opt.zero_grad()
        z = torch.randn(real.shape[0], NOISE_DIM)
        fake_y = torch.randint(0, num_classes, (real.shape[0],))
        fake = G(z, fake_y)
        g_loss = bce(D(fake, fake_y), ones)
        g_loss.backward()
        g_opt.step()

    return G, xmin, xmax


def generate(G, num_classes, xmin, xmax):
    samples, labels = [], []
    for cls in range(num_classes):
        z = torch.randn(AUG_PER_CLASS, NOISE_DIM)
        lab = torch.full((AUG_PER_CLASS,), cls, dtype=torch.long)
        with torch.no_grad():
            fake = G(z, lab).numpy()
        fake = ((fake + 1.0) / 2.0) * (xmax - xmin) + xmin
        fake = np.clip(fake, xmin, xmax)
        samples.append(fake)
        labels.append(np.full(AUG_PER_CLASS, cls, dtype=np.int64))
    return np.concatenate(samples, axis=0), np.concatenate(labels, axis=0)


def evaluate_with_gan(images, y, size, num_classes, binary):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    base_acc, aug_acc = [], []
    base_f1, aug_f1 = [], []
    base_auc, aug_auc = [], []
    base_ci, aug_ci = [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        base_pred = train_fullsize(images[tr], y[tr], images[te], size, binary)
        G, xmin, xmax = train_gan(images[tr], y[tr], num_classes, size)
        fake_x, fake_y = generate(G, num_classes, xmin, xmax)
        X_aug = np.concatenate([images[tr], fake_x], axis=0)
        y_aug = np.concatenate([y[tr], fake_y], axis=0)
        aug_pred = train_fullsize(X_aug, y_aug, images[te], size, binary)
        if binary:
            base_auc.append(roc_auc_score(y[te], base_pred))
            aug_auc.append(roc_auc_score(y[te], aug_pred))
            base_ci.append(concordance_index(y_time[te], -base_pred, y[te]))
            aug_ci.append(concordance_index(y_time[te], -aug_pred, y[te]))
        else:
            base_acc.append(accuracy_score(y[te], base_pred))
            aug_acc.append(accuracy_score(y[te], aug_pred))
            base_f1.append(f1_score(y[te], base_pred, average="macro"))
            aug_f1.append(f1_score(y[te], aug_pred, average="macro"))
    if binary:
        return {
            "baseline": {"roc_auc": base_auc, "c_index": base_ci},
            "gan_aug": {"roc_auc": aug_auc, "c_index": aug_ci},
        }
    return {
        "baseline": {"accuracy": base_acc, "macro_f1": base_f1},
        "gan_aug": {"accuracy": aug_acc, "macro_f1": aug_f1},
    }


def run_pam50():
    df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
    enc = LabelEncoder()
    y = enc.fit_transform(df["pam50"].values)
    imgs = np.load(IMG / "PAM50/mRNA/images.npy")
    res = evaluate_with_gan(imgs, y, 20, len(enc.classes_), binary=False)
    rows = []
    for variant, m in res.items():
        for metric in ["accuracy", "macro_f1"]:
            rows.append({"variant": variant, "task": "PAM50", "metric": metric, "mean": round(float(np.mean(m[metric])), 4), "std": round(float(np.std(m[metric])), 4)})
    return rows


def run_survival():
    global y_time
    df = pd.read_csv(SUR_DIR / "mRNA_Survival_final.tsv", sep="\t")
    y_event = df["os_event"].astype(int).values
    y_time = df["os_time_days"].astype(float).values
    imgs = np.load(IMG / "Survival/mRNA/images.npy")
    res = evaluate_with_gan(imgs, y_event, 15, 2, binary=True)
    rows = []
    for variant, m in res.items():
        for metric in ["roc_auc", "c_index"]:
            rows.append({"variant": variant, "task": "Survival", "metric": metric, "mean": round(float(np.mean(m[metric])), 4), "std": round(float(np.std(m[metric])), 4)})
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
