#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interpretability and key-factor mining pipeline."""

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
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PAM_DIR = DATA / "final_datasets/PAM50"
SUR_DIR = DATA / "final_datasets/Survival"
IMG = DATA / "images"
OUT_DIR = DATA / "interpretability"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import spiral_order  # noqa: E402


RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64


class FullSizeCNN(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, x):
        return self.head(F.relu(self.conv(x)))


def train_cnn(Xtr, ytr, Xte, size, out_dim):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeCNN(size, out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    n = yt.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte, dtype=torch.float32))
    return model, logits.argmax(dim=1).numpy()


def cnn_saliency(model, Xte):
    model.eval()
    x = torch.tensor(Xte, dtype=torch.float32, requires_grad=True)
    logits = model(x)
    pred = logits.argmax(dim=1)
    target = logits[torch.arange(len(pred)), pred].sum()
    model.zero_grad()
    target.backward()
    grads = x.grad.detach().abs().numpy()
    return grads


def main():
    df = pd.read_csv(PAM_DIR / "mRNA_PAM50_final.tsv", sep="\t")
    genes = df.columns[1:].tolist()
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    enc = LabelEncoder()
    y = enc.fit_transform(df["pam50"].values)
    images = np.load(IMG / "PAM50/mRNA/images.npy")
    order = pd.read_csv(IMG / "PAM50/mRNA/order.tsv", sep="\t")
    positions = spiral_order(20)
    pos_to_gene = {pos: gene for pos, gene in zip(positions[: len(order)], order["feature"].tolist())}

    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    shap_imp = {g: [] for g in genes}
    cnn_imp = {g: [] for g in genes}
    fold_top_lr = []
    fold_top_cnn = []

    for tr, te in cv.split(np.zeros(len(y)), y):
        # SHAP LogisticRegression
        model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        model.fit(X[tr], y[tr])
        explainer = shap.LinearExplainer(model, X[tr])
        shap_values = explainer.shap_values(X[te])
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            vals = np.abs(sv).mean(axis=(0, 2))
        else:
            vals = np.abs(sv).mean(axis=0)
        for i, g in enumerate(genes):
            shap_imp[g].append(float(vals[i]))
        top_lr = sorted(genes, key=lambda g: -np.mean(shap_imp[g]))[:100]
        fold_top_lr.append(top_lr)

        # CNN pixel saliency
        model, pred = train_cnn(images[tr], y[tr], images[te], 20, len(enc.classes_))
        grads = cnn_saliency(model, images[te])[:, 0]
        pixel_imp = grads.mean(axis=0)
        gene_sal = {}
        for pos, g in pos_to_gene.items():
            gene_sal[g] = gene_sal.get(g, 0.0) + float(pixel_imp[pos])
        for g in genes:
            cnn_imp[g].append(gene_sal.get(g, 0.0))
        top_cnn = sorted(genes, key=lambda g: -np.mean(cnn_imp[g]))[:100]
        fold_top_cnn.append(top_cnn)

    def summary(d):
        return pd.DataFrame(
            {
                "gene": list(d.keys()),
                "mean_importance": [float(np.mean(v)) for v in d.values()],
                "std_importance": [float(np.std(v)) for v in d.values()],
            }
        )
    shap_df = summary(shap_imp).sort_values("mean_importance", ascending=False)
    cnn_df = summary(cnn_imp).sort_values("mean_importance", ascending=False)
    def fold_freq(d, lists):
        freq = {g: 0 for g in genes}
        for top in lists:
            for g in set(top):
                freq[g] += 1
        d = d.copy()
        d["fold_freq_top100"] = d["gene"].map(freq)
        return d
    shap_df = fold_freq(shap_df, fold_top_lr)
    cnn_df = fold_freq(cnn_df, fold_top_cnn)

    shap_rank = shap_df.set_index("gene")["mean_importance"].rank(ascending=False)
    cnn_rank = cnn_df.set_index("gene")["mean_importance"].rank(ascending=False)
    consensus = pd.DataFrame({"gene": genes})
    consensus["shap_rank"] = consensus["gene"].map(shap_rank)
    consensus["cnn_rank"] = consensus["gene"].map(cnn_rank)
    consensus["mean_rank"] = (consensus["shap_rank"] + consensus["cnn_rank"]) / 2.0
    consensus = consensus.sort_values("mean_rank")

    shap_df.to_csv(OUT_DIR / "mrna_pam50_shap_importance.tsv", sep="\t", index=False)
    cnn_df.to_csv(OUT_DIR / "mrna_pam50_cnn_saliency_importance.tsv", sep="\t", index=False)
    consensus.to_csv(OUT_DIR / "mrna_pam50_consensus_key_factors.tsv", sep="\t", index=False)

    # Survival key factors: linear surrogate + univariate Cox
    sur = pd.read_csv(SUR_DIR / "mRNA_Survival_final.tsv", sep="\t")
    sur_X = sur.drop(columns=["case_id", "os_event", "os_time_days"]).to_numpy(dtype=float)
    sur_genes = sur.columns[3:].tolist()
    y_event = sur["os_event"].astype(int).values
    y_time = sur["os_time_days"].astype(float).values
    sur_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(sur_X)
    Xs = scaler.transform(sur_X)
    sur_model.fit(Xs, y_event)
    coef = np.abs(sur_model.coef_).ravel()
    sur_importance = pd.DataFrame({"gene": sur_genes, "lr_abs_coef": coef}).sort_values("lr_abs_coef", ascending=False)

    cox_rows = []
    for g in sur_importance.head(200)["gene"]:
        vals = sur[g].to_numpy(dtype=float)
        dfc = pd.DataFrame({"time": y_time, "event": y_event, "x": vals})
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(dfc, duration_col="time", event_col="event")
            cox_rows.append({"gene": g, "hr": float(np.exp(cph.params_.iloc[0])), "p": float(cph.summary["p"].iloc[0])})
        except Exception:
            cox_rows.append({"gene": g, "hr": np.nan, "p": np.nan})
    cox_df = pd.DataFrame(cox_rows)
    sur_importance = sur_importance.merge(cox_df, on="gene", how="left")
    sur_importance.to_csv(OUT_DIR / "survival_key_factors.tsv", sep="\t", index=False)

    print("Wrote interpretability outputs to", OUT_DIR, flush=True)
    print(shap_df.head(10).to_string(index=False))
    print(consensus.head(10).to_string(index=False))
    print(sur_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
