#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive integration - batch 1: single-omics ML / MLP / FullSizeCNN baselines."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent.parent
PAM_DIR = ROOT / "data/final_datasets/PAM50"
SUR_DIR = ROOT / "data/final_datasets/Survival"
IMG = ROOT / "data/images"
OUT = ROOT / "data/comprehensive_batch1_single_omics_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
OMICS = ["mRNA", "CNV", "miRNA"]


def ml_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVC": SVC(kernel="linear", probability=False, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


class MLP(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, dout),
        )

    def forward(self, x):
        return self.net(x)


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


def train_mlp(Xtr, ytr, Xte, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = MLP(Xtr.shape[1], 1 if binary else 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    if binary:
        crit = nn.BCEWithLogitsLoss()
        yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    else:
        crit = nn.CrossEntropyLoss()
        yt = torch.tensor(ytr, dtype=torch.long)
    n = yt.shape[0]
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
    if binary:
        return torch.sigmoid(out).numpy().ravel()
    return F.softmax(out, dim=1).numpy()


def train_cnn(Xtr, ytr, Xte, size, binary):
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
    n = yt.shape[0]
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
    if binary:
        return torch.sigmoid(out).numpy().ravel()
    return F.softmax(out, dim=1).numpy()


def risk_score(clf, Xte):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(Xte)[:, 1]
    return clf.decision_function(Xte)


def cox_cindex(Xtr, ytr, ytime_tr, Xte, ytime_te, yevent_te):
    variances = np.var(Xtr, axis=0)
    top_idx = np.argsort(variances)[::-1][:100]
    scaler = StandardScaler().fit(Xtr[:, top_idx])
    train_df = pd.DataFrame(scaler.transform(Xtr[:, top_idx]))
    test_df = pd.DataFrame(scaler.transform(Xte[:, top_idx]))
    train_df["time"] = ytime_tr
    train_df["event"] = ytr
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_df, duration_col="time", event_col="event")
        hazard = cph.predict_partial_hazard(test_df)
        return hazard
    except Exception:
        return np.zeros(Xte.shape[0])


def eval_pam50(omics, X, y, imgs, size):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {name: {"acc": [], "f1": []} for name in ml_models()}
    results["MLP"] = {"acc": [], "f1": []}
    results["FullSizeCNN"] = {"acc": [], "f1": []}
    for tr, te in cv.split(np.zeros(len(y)), y):
        for name, clf in ml_models().items():
            pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
            pipe.fit(X[tr], y[tr])
            pred = pipe.predict(X[te])
            results[name]["acc"].append(accuracy_score(y[te], pred))
            results[name]["f1"].append(f1_score(y[te], pred, average="macro"))
        p = train_mlp(X[tr], y[tr], X[te], binary=False)
        pred = p.argmax(axis=1)
        results["MLP"]["acc"].append(accuracy_score(y[te], pred))
        results["MLP"]["f1"].append(f1_score(y[te], pred, average="macro"))
        p = train_cnn(imgs[tr], y[tr], imgs[te], size, binary=False)
        pred = p.argmax(axis=1)
        results["FullSizeCNN"]["acc"].append(accuracy_score(y[te], pred))
        results["FullSizeCNN"]["f1"].append(f1_score(y[te], pred, average="macro"))
    rows = []
    for name, m in results.items():
        rows.append({"omics": omics, "task": "PAM50", "model": name, "metric": "accuracy", "mean": round(float(np.mean(m["acc"])), 4), "std": round(float(np.std(m["acc"])), 4)})
        rows.append({"omics": omics, "task": "PAM50", "model": name, "metric": "macro_f1", "mean": round(float(np.mean(m["f1"])), 4), "std": round(float(np.std(m["f1"])), 4)})
    return rows


def eval_survival(omics, X, y_event, y_time, imgs, size):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {name: {"auc": [], "ci": []} for name in ml_models()}
    results["MLP"] = {"auc": [], "ci": []}
    results["FullSizeCNN"] = {"auc": [], "ci": []}
    results["CoxPH"] = {"auc": [], "ci": []}
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        for name, clf in ml_models().items():
            pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
            pipe.fit(X[tr], y_event[tr])
            prob = risk_score(pipe[-1], pipe[0].transform(X[te]))
            results[name]["auc"].append(roc_auc_score(y_event[te], prob))
            results[name]["ci"].append(concordance_index(y_time[te], -prob, y_event[te]))
        prob = train_mlp(X[tr], y_event[tr], X[te], binary=True)
        results["MLP"]["auc"].append(roc_auc_score(y_event[te], prob))
        results["MLP"]["ci"].append(concordance_index(y_time[te], -prob, y_event[te]))
        prob = train_cnn(imgs[tr], y_event[tr], imgs[te], size, binary=True)
        results["FullSizeCNN"]["auc"].append(roc_auc_score(y_event[te], prob))
        results["FullSizeCNN"]["ci"].append(concordance_index(y_time[te], -prob, y_event[te]))
        hazard = cox_cindex(X[tr], y_event[tr], y_time[tr], X[te], y_time[te], y_event[te])
        results["CoxPH"]["auc"].append(roc_auc_score(y_event[te], hazard))
        results["CoxPH"]["ci"].append(concordance_index(y_time[te], -hazard, y_event[te]))
    rows = []
    for name, m in results.items():
        rows.append({"omics": omics, "task": "Survival", "model": name, "metric": "roc_auc", "mean": round(float(np.mean(m["auc"])), 4), "std": round(float(np.std(m["auc"])), 4)})
        rows.append({"omics": omics, "task": "Survival", "model": name, "metric": "c_index", "mean": round(float(np.mean(m["ci"])), 4), "std": round(float(np.std(m["ci"])), 4)})
    return rows


def main():
    size_pam = {"mRNA": 20, "CNV": 8, "miRNA": 25}
    size_sur = {"mRNA": 15, "CNV": 13, "miRNA": 25}
    rows = []
    for omics in OMICS:
        df = pd.read_csv(PAM_DIR / f"{omics}_PAM50_final.tsv", sep="\t")
        X = df.drop(columns=["pam50"]).to_numpy(dtype=float)
        enc = LabelEncoder()
        y = enc.fit_transform(df["pam50"].values)
        imgs = np.load(IMG / "PAM50" / omics / "images.npy")
        rows += eval_pam50(omics, X, y, imgs, size_pam[omics])
        print(f"done PAM50 {omics}", flush=True)

    for omics in OMICS:
        df = pd.read_csv(SUR_DIR / f"{omics}_Survival_final.tsv", sep="\t")
        X = df.drop(columns=["case_id", "os_event", "os_time_days"]).to_numpy(dtype=float)
        y_event = df["os_event"].astype(int).values
        y_time = df["os_time_days"].astype(float).values
        imgs = np.load(IMG / "Survival" / omics / "images.npy")
        rows += eval_survival(omics, X, y_event, y_time, imgs, size_sur[omics])
        print(f"done Survival {omics}", flush=True)

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["omics", "task", "model", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
