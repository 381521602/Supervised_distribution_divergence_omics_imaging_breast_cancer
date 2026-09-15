#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple deep-learning baselines with PyTorch for PAM50 and survival."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "deep_learning_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 50
BATCH_SIZE = 64
DEVICE = torch.device("cpu")


def load_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def build_fusion(task: str, omics: list[str]):
    matrices = [load_matrix(FEAT_DIR / f"{o}_{task}_matrix.tsv") for o in omics]
    common = matrices[0].index
    for m in matrices[1:]:
        common = common.intersection(m.index)
    common = list(common)
    X = np.hstack([m.loc[common].to_numpy(dtype=float) for m in matrices])
    return X, np.array(common)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def torch_seed():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def train_model(model, X_train, y_train, task, epochs=EPOCHS):
    torch_seed()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if task == "PAM50_4class":
        criterion = nn.CrossEntropyLoss()
        y = torch.tensor(y_train, dtype=torch.long)
    else:
        criterion = nn.BCEWithLogitsLoss()
        y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X = torch.tensor(X_train, dtype=torch.float32)

    model.train()
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            out = model(X[idx])
            loss = criterion(out, y[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))
    return out


def cv_pam50(fusion: str, config: str, hidden: list[int], X, y) -> list[dict]:
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for train_idx, test_idx in cv.split(X, y_enc):
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = MLP(X_train.shape[1], hidden, len(encoder.classes_))
        model = train_model(model, X_train, y_enc[train_idx], "PAM50_4class")
        probs = torch.softmax(predict(model, X_test), dim=1).numpy()
        pred = probs.argmax(axis=1)
        accs.append(accuracy_score(y_enc[test_idx], pred))
        f1s.append(f1_score(y_enc[test_idx], pred, average="macro"))
    return [
        {"fusion": fusion, "task": "PAM50_4class", "model": config, "metric": "accuracy", "value": round(float(np.mean(accs)), 4)},
        {"fusion": fusion, "task": "PAM50_4class", "model": config, "metric": "macro_f1", "value": round(float(np.mean(f1s)), 4)},
    ]


def cv_binary(fusion: str, config: str, hidden: list[int], X, y) -> list[dict]:
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = MLP(X_train.shape[1], hidden, 1)
        model = train_model(model, X_train, y[train_idx], "OS_binary")
        probs = torch.sigmoid(predict(model, X_test)).numpy().ravel()
        aucs.append(roc_auc_score(y[test_idx], probs))
    return [
        {"fusion": fusion, "task": "OS_binary", "model": config, "metric": "roc_auc", "value": round(float(np.mean(aucs)), 4)}
    ]


def cox_loss(log_h, time, event):
    risk = log_h.view(-1)
    _, idx = torch.sort(time, descending=True)
    risk = risk[idx]
    event = event[idx]
    exp_risk = torch.exp(risk)
    cumsum = torch.cumsum(exp_risk, dim=0)
    log_denom = torch.log(cumsum + 1e-8)
    return -torch.mean((risk - log_denom) * event.float())


def cv_cox(fusion: str, config: str, hidden: list[int], X, y_time, y_event) -> list[dict]:
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    c_indices = []
    for train_idx, test_idx in cv.split(X, y_event):
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = MLP(X_train.shape[1], hidden, 1)
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.tensor(X_train, dtype=torch.float32)
        time_t = torch.tensor(y_time[train_idx], dtype=torch.float32)
        event_t = torch.tensor(y_event[train_idx], dtype=torch.float32)
        model.train()
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            log_h = model(Xt)
            loss = cox_loss(log_h, time_t, event_t)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            hazard = model(torch.tensor(X_test, dtype=torch.float32)).numpy().ravel()
        c_indices.append(float(concordance_index(y_time[test_idx], -hazard, y_event[test_idx])))
    return [
        {"fusion": fusion, "task": "OS_Cox", "model": config, "metric": "c_index", "value": round(float(np.mean(c_indices)), 4)}
    ]


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    rows: list[dict] = []
    configs = {
        "MLP_128_64": [128, 64],
        "MLP_256_128_64": [256, 128, 64],
    }

    fusions = [
        ("mRNA+CNV+miRNA", ["mRNA", "CNV", "miRNA"]),
        ("mRNA+CNV", ["mRNA", "CNV"]),
    ]
    for fusion_name, omics in fusions:
        for task in ["PAM50_4class", "OS"]:
            X, cases = build_fusion(task, omics)
            if task == "PAM50_4class":
                y = labels.set_index("case_id").loc[cases, "pam50_4class"].values
                for config, hidden in configs.items():
                    rows.extend(cv_pam50(fusion_name, config, hidden, X, y))
            else:
                y_event = labels.set_index("case_id").loc[cases, "os_event"].astype(int).values
                y_time = labels.set_index("case_id").loc[cases, "os_time_days"].astype(float).values
                for config, hidden in configs.items():
                    rows.extend(cv_binary(fusion_name, config, hidden, X, y_event))
                    rows.extend(cv_cox(fusion_name, config, hidden, X, y_time, y_event))

    if rows:
        columns = ["fusion", "task", "model", "metric", "value"]
        with OUT.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
