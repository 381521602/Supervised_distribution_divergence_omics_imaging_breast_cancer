# -*- coding: utf-8 -*-
"""Perturbation-fidelity comparison of attribution rankings for mRNA PAM50."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(r"E:\组学数据图像化")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "github_release" / "scripts"))

from run_full_s5_s7 import (
    align, fold_sel, jsd_scores, load_omics, make_images,
    OMICS, PAM50_K, PAM50_SIZE, RANDOM_STATE,
)
from run_multistream_cnn import spiral_order

OUT_TSV = ROOT / "data" / "perturbation_fidelity_results.tsv"
OUT_FIG = ROOT / "data" / "paper_figures" / "fig_perturbation_fidelity.png"
FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_RANDOM = 10
EPOCHS = 30
BATCH = 64


class FullSizeCNN(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, x):
        return self.head(F.relu(self.conv(x)))


def train_cnn_model(itr, ytr, size, out_dim):
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    model = FullSizeCNN(size, out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(itr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    n = yt.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
    model.eval()
    return model


def cnn_saliency_ranking(model, itr, order, n_feat, size):
    model.eval()
    x = torch.tensor(itr, dtype=torch.float32, requires_grad=True)
    logits = model(x)
    pred = logits.argmax(dim=1)
    target = logits[torch.arange(len(pred)), pred].sum()
    model.zero_grad()
    target.backward()
    grads = x.grad.detach().abs().numpy()[:, 0]
    pixel_imp = grads.mean(axis=0)
    positions = spiral_order(size)
    feat_sal = np.zeros(n_feat, dtype=float)
    for (r, c), p in zip(positions, range(len(positions))):
        if p < n_feat:
            feat_sal[order[p]] += float(pixel_imp[r, c])
    return np.argsort(feat_sal)[::-1]


def accuracy_at(Xte, yte, clf, ranking, frac):
    k = int(round(frac * Xte.shape[1]))
    Xp = Xte.copy()
    if k > 0:
        Xp[:, ranking[:k]] = 0.0
    return accuracy_score(yte, clf.predict(Xp))


def cnn_accuracy_at(ite, yte, model, ranking, frac, order, size, n_feat):
    k = int(round(frac * n_feat))
    mask = np.zeros(n_feat, dtype=bool)
    if k > 0:
        mask[ranking[:k]] = True
    positions = spiral_order(size)
    itp = ite.copy()
    for (r, c), p in zip(positions, range(len(positions))):
        if p < n_feat and mask[order[p]]:
            itp[:, 0, r, c] = 0.0
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(itp, dtype=torch.float32))
    return accuracy_score(yte, logits.argmax(dim=1).numpy())


def main():
    data, labels = load_omics("PAM50_4class")
    X, y, _, cases = align(data, labels, "PAM50_4class", OMICS)
    k_feat = PAM50_K["mRNA"]
    size = PAM50_SIZE["mRNA"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = ["LogisticRegression", "FullSizeCNN"]
    rankings = ["JSD", "SHAP", "Saliency", "Random"]
    curves = {m: {r: {f: [] for f in FRACTIONS} for r in rankings} for m in models}

    for tr, te in cv.split(np.zeros(len(y)), y):
        pipe = fold_sel(X, y, "PAM50_4class", "mRNA", tr)
        Xtr = pipe.transform(X["mRNA"][tr])
        Xte = pipe.transform(X["mRNA"][te])
        order = np.argsort(jsd_scores(Xtr, y[tr]))[::-1]
        itr = make_images(Xtr[:, order], np.arange(Xtr.shape[1]), size)
        ite = make_images(Xte[:, order], np.arange(Xte.shape[1]), size)

        lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(Xtr, y[tr])
        expl = shap.LinearExplainer(lr, Xtr)
        sv = np.asarray(expl.shap_values(Xtr))
        if sv.ndim == 3:
            shap_imp = np.abs(sv).mean(axis=(0, 2))
        else:
            shap_imp = np.abs(sv).mean(axis=0)
        ranking = {
            "JSD": np.argsort(jsd_scores(Xtr, y[tr]))[::-1],
            "SHAP": np.argsort(shap_imp)[::-1],
        }

        cnn = train_cnn_model(itr, y[tr], size, 4)
        ranking["Saliency"] = cnn_saliency_ranking(cnn, itr, order, Xtr.shape[1], size)
        rng = np.random.default_rng(RANDOM_STATE)
        rand_rankings = [rng.permutation(Xtr.shape[1]) for _ in range(N_RANDOM)]

        for f in FRACTIONS:
            curves["LogisticRegression"]["JSD"][f].append(
                accuracy_at(Xte, y[te], lr, ranking["JSD"], f))
            curves["LogisticRegression"]["SHAP"][f].append(
                accuracy_at(Xte, y[te], lr, ranking["SHAP"], f))
            curves["LogisticRegression"]["Random"][f].append(
                float(np.mean([accuracy_at(Xte, y[te], lr, rr, f) for rr in rand_rankings])))
            curves["FullSizeCNN"]["JSD"][f].append(
                cnn_accuracy_at(ite, y[te], cnn, ranking["JSD"], f, order, size, Xtr.shape[1]))
            curves["FullSizeCNN"]["Saliency"][f].append(
                cnn_accuracy_at(ite, y[te], cnn, ranking["Saliency"], f, order, size, Xtr.shape[1]))
            curves["FullSizeCNN"]["Random"][f].append(
                float(np.mean([cnn_accuracy_at(ite, y[te], cnn, rr, f, order, size, Xtr.shape[1])
                               for rr in rand_rankings])))
        print("fold done", flush=True)

    rows = []
    for m in models:
        for r in rankings:
            for f in FRACTIONS:
                vals = curves[m][r][f]
                rows.append([m, r, f, round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)])
    with OUT_TSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["model", "ranking", "fraction_removed", "mean_accuracy", "std_accuracy"])
        w.writerows(rows)
    print("wrote", OUT_TSV)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(OUT_TSV, sep="\t")
    colors = {"JSD": "#C0504D", "SHAP": "#2E74B5", "Saliency": "#0E7C86", "Random": "#8A8A8A"}
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 11, "axes.edgecolor": "#444444", "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, m, title in [(axes[0], "LogisticRegression", "A. LogisticRegression"),
                         (axes[1], "FullSizeCNN", "B. FullSizeCNN")]:
        sub = df[df["model"] == m]
        for r in rankings:
            d = sub[sub["ranking"] == r].sort_values("fraction_removed")
            ax.plot(d["fraction_removed"] * 100, d["mean_accuracy"], "-o", ms=3,
                    color=colors[r], label=r, lw=1.6)
        ax.set_xlabel("Top features removed (%)", fontweight="bold")
        ax.set_ylabel("Accuracy", fontweight="bold")
        ax.set_title(title, loc="left", fontweight="bold", color="#1F4D78", fontsize=12.5)
        ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0.15, 0.9)
    axes[0].legend(fontsize=9.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)
    plt.close(fig)
    print("wrote", OUT_FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
