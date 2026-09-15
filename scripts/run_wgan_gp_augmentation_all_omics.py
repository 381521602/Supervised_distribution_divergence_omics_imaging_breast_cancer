#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WGAN-GP 扩增三组学单组学 FullSizeCNN。

每个组学、任务、折内：
1. 只用训练折训练一个 WGAN-GP；
2. 从该生成器分别生成 1x/5x/10x 样本；
3. 分别训练 FullSizeCNN 并在测试折评估。
测试折不参与 GAN、标准化或扩增。
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
OUT = ROOT / "data/wgan_gp_augmentation_all_omics_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
NOISE_DIM = 64
EMBED_DIM = 16
GAN_ITERS = 250
CRITIC_ITERS = 3
GP_WEIGHT = 10.0
PER_CLASS = {"1x": 120, "5x": 600, "10x": 1200}


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


class Critic(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Linear(64 + embed_dim, 1)

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


def gradient_penalty(D, real, fake, labels):
    b, c, h, w = real.shape
    alpha = torch.rand(b, 1, 1, 1)
    interp = alpha * real + (1.0 - alpha) * fake
    interp = interp.requires_grad_(True)
    out = D(interp, labels)
    grads = torch.autograd.grad(
        outputs=out,
        inputs=interp,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((grads.view(b, -1).norm(2, dim=1) - 1.0) ** 2).mean()


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


def train_wgan_gp(images, labels, num_classes, output_size):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    xmin = float(np.min(images))
    xmax = float(np.max(images))
    x_norm = ((images - xmin) / max(xmax - xmin, 1e-6)) * 2.0 - 1.0
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    G = Generator(NOISE_DIM, EMBED_DIM, num_classes, output_size)
    D = Critic(EMBED_DIM, num_classes)
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
    n = x_tensor.shape[0]

    for _ in range(GAN_ITERS):
        for _ in range(CRITIC_ITERS):
            idx = torch.randint(0, n, (BATCH_SIZE,))
            real = x_tensor[idx]
            real_y = y_tensor[idx]
            z = torch.randn(real.shape[0], NOISE_DIM)
            fake_y = torch.randint(0, num_classes, (real.shape[0],))
            fake = G(z, fake_y).detach()
            gp = gradient_penalty(D, real, fake, real_y)
            d_loss = -D(real, real_y).mean() + D(fake, fake_y).mean() + GP_WEIGHT * gp
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

        z = torch.randn(BATCH_SIZE, NOISE_DIM)
        fake_y = torch.randint(0, num_classes, (BATCH_SIZE,))
        fake = G(z, fake_y)
        g_loss = -D(fake, fake_y).mean()
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

    return G, xmin, xmax


def generate(G, num_classes, xmin, xmax, per_class):
    samples, labels = [], []
    for cls in range(num_classes):
        z = torch.randn(per_class, NOISE_DIM)
        lab = torch.full((per_class,), cls, dtype=torch.long)
        with torch.no_grad():
            fake = G(z, lab).numpy()
        fake = ((fake + 1.0) / 2.0) * (xmax - xmin) + xmin
        fake = np.clip(fake, xmin, xmax)
        samples.append(fake)
        labels.append(np.full(per_class, cls, dtype=np.int64))
    return np.concatenate(samples, axis=0), np.concatenate(labels, axis=0)


def evaluate_omics_task(task, omics, size, y, num_classes, binary, y_time=None):
    images = np.load(IMG / task / omics / "images.npy")
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metrics = {
        "baseline": {"accuracy": [], "macro_f1": [], "roc_auc": [], "c_index": []},
        "aug_1x": {"accuracy": [], "macro_f1": [], "roc_auc": [], "c_index": []},
        "aug_5x": {"accuracy": [], "macro_f1": [], "roc_auc": [], "c_index": []},
        "aug_10x": {"accuracy": [], "macro_f1": [], "roc_auc": [], "c_index": []},
    }
    for tr, te in cv.split(np.zeros(len(y)), y):
        base_pred = train_fullsize(images[tr], y[tr], images[te], size, binary)
        G, xmin, xmax = train_wgan_gp(images[tr], y[tr], num_classes, size)
        preds = {"baseline": base_pred}
        for key, per_class in PER_CLASS.items():
            fake_x, fake_y = generate(G, num_classes, xmin, xmax, per_class)
            X_aug = np.concatenate([images[tr], fake_x], axis=0)
            y_aug = np.concatenate([y[tr], fake_y], axis=0)
            preds["aug_" + key] = train_fullsize(X_aug, y_aug, images[te], size, binary)
        for key, pred in preds.items():
            if binary:
                metrics[key]["roc_auc"].append(roc_auc_score(y[te], pred))
                metrics[key]["c_index"].append(concordance_index(y_time[te], -pred, y[te]))
            else:
                metrics[key]["accuracy"].append(accuracy_score(y[te], pred))
                metrics[key]["macro_f1"].append(f1_score(y[te], pred, average="macro"))
    return metrics


def load_pam50(omics):
    df = pd.read_csv(PAM_DIR / f"{omics}_PAM50_final.tsv", sep="\t")
    enc = LabelEncoder()
    y = enc.fit_transform(df["pam50"].values)
    return y, len(enc.classes_)


def load_survival(omics):
    df = pd.read_csv(SUR_DIR / f"{omics}_Survival_final.tsv", sep="\t")
    y_event = df["os_event"].astype(int).values
    y_time = df["os_time_days"].astype(float).values
    return y_event, y_time


def main():
    omics_list = ["mRNA", "CNV", "miRNA"]
    size_pam = {"mRNA": 20, "CNV": 8, "miRNA": 25}
    size_sur = {"mRNA": 15, "CNV": 13, "miRNA": 25}
    rows = []
    for omics in omics_list:
        y, n_classes = load_pam50(omics)
        metrics = evaluate_omics_task("PAM50", omics, size_pam[omics], y, n_classes, binary=False)
        for key in ["baseline", "aug_1x", "aug_5x", "aug_10x"]:
            for metric in ["accuracy", "macro_f1"]:
                rows.append({"omics": omics, "task": "PAM50", "variant": key, "metric": metric, "mean": round(float(np.mean(metrics[key][metric])), 4), "std": round(float(np.std(metrics[key][metric])), 4)})
        print(f"done PAM50 {omics}", flush=True)

    for omics in omics_list:
        y_event, y_time = load_survival(omics)
        metrics = evaluate_omics_task("Survival", omics, size_sur[omics], y_event, 2, binary=True, y_time=y_time)
        for key in ["baseline", "aug_1x", "aug_5x", "aug_10x"]:
            for metric in ["roc_auc", "c_index"]:
                rows.append({"omics": omics, "task": "Survival", "variant": key, "metric": metric, "mean": round(float(np.mean(metrics[key][metric])), 4), "std": round(float(np.std(metrics[key][metric])), 4)})
        print(f"done Survival {omics}", flush=True)

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["omics", "task", "variant", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)
    for r in rows:
        print(r["omics"], r["task"], r["variant"], r["metric"], r["mean"], "+-", r["std"])


if __name__ == "__main__":
    main()
