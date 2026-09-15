#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare methods per omics on each omics' maximum available samples."""

from __future__ import annotations

import csv
import sys
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
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import feature_order, to_images_with_order  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "single_omics_methods_max_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 50
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
OMICS = ["mRNA", "CNV", "miRNA"]
SIZES = {"mRNA": 15, "CNV": 8, "miRNA": 15}


def load_matrix(task: str, omics: str) -> pd.DataFrame:
    return pd.read_csv(FEAT_DIR / f"{omics}_{task}_matrix.tsv", sep="\t", index_col=0)


def vector_classifiers():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVC": SVC(kernel="linear", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class FullSizeCNN(nn.Module):
    def __init__(self, size, out_dim, channels=32):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size=(size, size))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(channels, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim),
        )
    def forward(self, x):
        return self.head(self.conv(x))


def torch_seed():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def train_mlp(X_train, y_train, X_test, y_test, out_dim):
    torch_seed()
    model = MLP(X_train.shape[1], out_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    n = Xt.shape[0]
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
    return logits.argmax(dim=1).numpy()


def train_cnn(X_train, y_train, X_test, y_test, size, out_dim):
    torch_seed()
    model = FullSizeCNN(size, out_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    n = Xt.shape[0]
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
    return logits.argmax(dim=1).numpy()


def eval_pam50(omics: str):
    matrix = load_matrix("PAM50_4class", omics)
    cases = list(matrix.index)
    X = matrix.loc[cases].to_numpy(dtype=float)
    y_raw = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str}).set_index("case_id").loc[cases, "pam50_4class"].values
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, clf in vector_classifiers().items():
        model = Pipeline([("scale", StandardScaler()), ("clf", clf)])
        accs, f1s = [], []
        for tr, te in cv.split(np.zeros(len(cases)), y):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="macro"))
        rows.append({"omics": omics, "task": "PAM50_4class", "method": name, "metric": "accuracy", "value": round(float(np.mean(accs)), 4)})
        rows.append({"omics": omics, "task": "PAM50_4class", "method": name, "metric": "macro_f1", "value": round(float(np.mean(f1s)), 4)})
    # MLP
    accs, f1s = [], []
    for tr, te in cv.split(np.zeros(len(cases)), y):
        scaler = StandardScaler().fit(X[tr])
        pred = train_mlp(scaler.transform(X[tr]), y[tr], scaler.transform(X[te]), y[te], len(enc.classes_))
        accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
    rows.append({"omics": omics, "task": "PAM50_4class", "method": "MLP", "metric": "accuracy", "value": round(float(np.mean(accs)), 4)})
    rows.append({"omics": omics, "task": "PAM50_4class", "method": "MLP", "metric": "macro_f1", "value": round(float(np.mean(f1s)), 4)})
    # Full-size CNN
    size = SIZES[omics]
    accs, f1s = [], []
    for tr, te in cv.split(np.zeros(len(cases)), y):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
        order = feature_order(Xtr, y[tr])
        Xtr_img = to_images_with_order(Xtr, size, order)
        Xte_img = to_images_with_order(Xte, size, order)
        pred = train_cnn(Xtr_img, y[tr], Xte_img, y[te], size, len(enc.classes_))
        accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
    rows.append({"omics": omics, "task": "PAM50_4class", "method": "FullSizeCNN", "metric": "accuracy", "value": round(float(np.mean(accs)), 4)})
    rows.append({"omics": omics, "task": "PAM50_4class", "method": "FullSizeCNN", "metric": "macro_f1", "value": round(float(np.mean(f1s)), 4)})
    return rows


def eval_os(omics: str):
    matrix = load_matrix("OS", omics)
    cases = list(matrix.index)
    X = matrix.loc[cases].to_numpy(dtype=float)
    lab = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str}).set_index("case_id").loc[cases]
    y_event = lab["os_event"].astype(int).values
    y_time = lab["os_time_days"].astype(float).values
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
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
        rows.append({"omics": omics, "task": "OS_binary", "method": name, "metric": "roc_auc", "value": round(float(np.mean(aucs)), 4)})
    # Cox top variance
    c_index = []
    for tr, te in cv.split(np.zeros(len(cases)), y_event):
        var = np.var(X[tr], axis=0)
        top = np.argsort(var)[::-1][:100]
        scaler = StandardScaler().fit(X[tr][:, top])
        train_df = pd.DataFrame(scaler.transform(X[tr][:, top])); train_df["time"] = y_time[tr]; train_df["event"] = y_event[tr]
        test_df = pd.DataFrame(scaler.transform(X[te][:, top]))
        cph = CoxPHFitter(penalizer=0.1); cph.fit(train_df, duration_col="time", event_col="event")
        hazard = cph.predict_partial_hazard(test_df)
        c_index.append(concordance_index(y_time[te], -hazard, y_event[te]))
    rows.append({"omics": omics, "task": "OS_Cox", "method": "CoxPH", "metric": "c_index", "value": round(float(np.mean(c_index)), 4)})
    return rows


def main() -> None:
    rows = []
    for omics in OMICS:
        rows.extend(eval_pam50(omics))
        rows.extend(eval_os(omics))
    columns = ["omics", "task", "method", "metric", "value"]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
