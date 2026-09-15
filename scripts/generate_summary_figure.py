#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect 5-fold means/std and produce summary figures."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_single_omics_methods_max import (  # noqa: E402
    FullSizeCNN,
    MLP,
    OMICS,
    SIZES,
    vector_classifiers,
    load_matrix,
)
from run_multistream_cnn import feature_order, to_images_with_order  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
OUT_CSV = DATA / "summary_metrics_with_std.tsv"
FIG_DIR = DATA / "figures"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 64


import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


def torch_seed():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def train_mlp_fold(Xtr, ytr, Xte, yte, out_dim):
    torch_seed()
    model = MLP(Xtr.shape[1], out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    n = Xt.shape[0]
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad(); loss = crit(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte, dtype=torch.float32))
    return logits.argmax(dim=1).numpy()


def train_cnn_fold(Xtr, ytr, Xte, yte, size, out_dim):
    torch_seed()
    model = FullSizeCNN(size, out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    n = Xt.shape[0]
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad(); loss = crit(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte, dtype=torch.float32))
    return logits.argmax(dim=1).numpy()


def collect_pam50():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    records = []
    for omics in OMICS:
        matrix = load_matrix("PAM50_4class", omics)
        cases = list(matrix.index)
        X = matrix.loc[cases].to_numpy(dtype=float)
        y_raw = labels.set_index("case_id").loc[cases, "pam50_4class"].values
        enc = LabelEncoder(); y = enc.fit_transform(y_raw)
        cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for name, clf in vector_classifiers().items():
            model = Pipeline([("scale", StandardScaler()), ("clf", clf)])
            accs, f1s = [], []
            for tr, te in cv.split(np.zeros(len(cases)), y):
                model.fit(X[tr], y[tr]); pred = model.predict(X[te])
                accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
            records.append({"omics": omics, "task": "PAM50_4class", "method": name, "metric": "accuracy", "mean": np.mean(accs), "std": np.std(accs)})
            records.append({"omics": omics, "task": "PAM50_4class", "method": name, "metric": "macro_f1", "mean": np.mean(f1s), "std": np.std(f1s)})
        accs, f1s = [], []
        for tr, te in cv.split(np.zeros(len(cases)), y):
            scaler = StandardScaler().fit(X[tr])
            pred = train_mlp_fold(scaler.transform(X[tr]), y[tr], scaler.transform(X[te]), y[te], len(enc.classes_))
            accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
        records.append({"omics": omics, "task": "PAM50_4class", "method": "MLP", "metric": "accuracy", "mean": np.mean(accs), "std": np.std(accs)})
        records.append({"omics": omics, "task": "PAM50_4class", "method": "MLP", "metric": "macro_f1", "mean": np.mean(f1s), "std": np.std(f1s)})
        accs, f1s = [], []
        size = SIZES[omics]
        for tr, te in cv.split(np.zeros(len(cases)), y):
            scaler = StandardScaler().fit(X[tr]); Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
            order = feature_order(Xtr, y[tr])
            pred = train_cnn_fold(to_images_with_order(Xtr, size, order), y[tr], to_images_with_order(Xte, size, order), y[te], size, len(enc.classes_))
            accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
        records.append({"omics": omics, "task": "PAM50_4class", "method": "FullSizeCNN", "metric": "accuracy", "mean": np.mean(accs), "std": np.std(accs)})
        records.append({"omics": omics, "task": "PAM50_4class", "method": "FullSizeCNN", "metric": "macro_f1", "mean": np.mean(f1s), "std": np.std(f1s)})
    return pd.DataFrame(records)


def collect_os():
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    records = []
    for omics in OMICS:
        matrix = load_matrix("OS", omics); cases = list(matrix.index)
        X = matrix.loc[cases].to_numpy(dtype=float)
        lab = labels.set_index("case_id").loc[cases]
        y_event = lab["os_event"].astype(int).values; y_time = lab["os_time_days"].astype(float).values
        cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for name, clf in vector_classifiers().items():
            model = Pipeline([("scale", StandardScaler()), ("clf", clf)])
            aucs = []
            for tr, te in cv.split(np.zeros(len(cases)), y_event):
                model.fit(X[tr], y_event[tr])
                if hasattr(model[-1], "predict_proba"):
                    prob = model.predict_proba(X[te])[:, 1]
                else:
                    prob = model.decision_function(X[te])
                aucs.append(roc_auc_score(y_event[te], prob))
            records.append({"omics": omics, "task": "OS_binary", "method": name, "metric": "roc_auc", "mean": np.mean(aucs), "std": np.std(aucs)})
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
        cis = []
        for tr, te in cv.split(np.zeros(len(cases)), y_event):
            var = np.var(X[tr], axis=0); top = np.argsort(var)[::-1][:100]
            scaler = StandardScaler().fit(X[tr][:, top])
            train_df = pd.DataFrame(scaler.transform(X[tr][:, top])); train_df["time"] = y_time[tr]; train_df["event"] = y_event[tr]
            test_df = pd.DataFrame(scaler.transform(X[te][:, top]))
            cph = CoxPHFitter(penalizer=0.1); cph.fit(train_df, duration_col="time", event_col="event")
            cis.append(concordance_index(y_time[te], -cph.predict_partial_hazard(test_df), y_event[te]))
        records.append({"omics": omics, "task": "OS_Cox", "method": "CoxPH", "metric": "c_index", "mean": np.mean(cis), "std": np.std(cis)})
    return pd.DataFrame(records)


def plot_pam50(df):
    for metric in ["accuracy", "macro_f1"]:
        sub = df[(df["task"] == "PAM50_4class") & (df["metric"] == metric)]
        methods = ["LogisticRegression", "GradientBoosting", "MLP", "FullSizeCNN"]
        sub = sub[sub["method"].isin(methods)]
        fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=160)
        x = np.arange(len(OMICS)); w = 0.2
        for i, method in enumerate(methods):
            vals = [sub[(sub.omics == o) & (sub.method == method)]["mean"].values[0] for o in OMICS]
            stds = [sub[(sub.omics == o) & (sub.method == method)]["std"].values[0] for o in OMICS]
            ax.bar(x + (i - 1.5) * w, vals, w, yerr=stds, capsize=3, label=method)
        ax.set_xticks(x); ax.set_xticklabels(OMICS)
        ax.set_ylim(0, 1); ax.set_ylabel(metric)
        ax.set_title(f"PAM50 4-class: {metric} (mean +/- std)", loc="left", fontweight="bold")
        ax.legend(frameon=False, ncol=4); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(FIG_DIR / f"summary_pam50_{metric}.png", bbox_inches="tight"); plt.close(fig)


def plot_os(df):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=160)
    for ax, metric, task in [(axes[0], "roc_auc", "OS_binary"), (axes[1], "c_index", "OS_Cox")]:
        sub = df[(df["task"] == task) & (df["metric"] == metric)]
        methods = ["CoxPH"] if task == "OS_Cox" else ["LogisticRegression", "GradientBoosting"]
        sub = sub[sub["method"].isin(methods)]
        x = np.arange(len(OMICS)); w = 0.35
        for i, method in enumerate(methods):
            vals = [sub[(sub.omics == o) & (sub.method == method)]["mean"].values[0] for o in OMICS]
            stds = [sub[(sub.omics == o) & (sub.method == method)]["std"].values[0] for o in OMICS]
            ax.bar(x + (i - 0.5) * w, vals, w, yerr=stds, capsize=3, label=method)
        ax.set_xticks(x); ax.set_xticklabels(OMICS)
        ax.set_ylabel(metric); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(metric, loc="left", fontweight="bold")
    fig.suptitle("Survival: mean +/- std", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG_DIR / "summary_survival.png", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    pam = collect_pam50()
    os_df = collect_os()
    all_df = pd.concat([pam, os_df], ignore_index=True)
    all_df.to_csv(OUT_CSV, sep="\t", index=False)
    plot_pam50(all_df)
    plot_os(all_df)
    print(f"Saved -> {OUT_CSV}")
    print("Figures -> summary_pam50_accuracy.png, summary_pam50_macro_f1.png, summary_survival.png")


if __name__ == "__main__":
    main()
