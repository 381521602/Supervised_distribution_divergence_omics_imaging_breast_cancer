# -*- coding: utf-8 -*-
"""Rerun CNV single-omics with the corrected GDC CNV matrix."""
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(r"E:\组学数据图像化")
D = ROOT / "data"
GH = ROOT / "github_release"
sys.path.insert(0, str(GH / "scripts"))
from adaptive_jsd import build_adaptive_pipeline, jsd_between_histograms
from run_multistream_cnn import spiral_order

LABELS = GH / "data" / "brca_labels_modeling_ready.tsv"
OUT = ROOT / "data" / "leakfree_cnv_single_rerun.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
PAM50_K = 50
SURV_K = 150
PAM50_SIZE = 8
SURV_SIZE = 13


class MLP(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, dout))
    def forward(self, x):
        return self.net(x)


class FullSizeCNN(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))
    def forward(self, x):
        return self.head(F.relu(self.conv(x)))


class Dense(nn.Module):
    def __init__(self, din, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))
    def forward(self, x):
        return self.net(x)


def jsd_scores(X, y, n_bins=10):
    classes = np.unique(y)
    scores = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        feat = X[:, j]
        finite = feat[np.isfinite(feat)]
        if finite.size < 10:
            continue
        qs = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)))
        if qs.size < 2:
            continue
        bins = np.digitize(feat, qs[1:-1])
        hists = [np.bincount(bins[y == c], minlength=n_bins) for c in classes]
        if len(hists) == 2:
            scores[j] = jsd_between_histograms(hists[0], hists[1])
        else:
            ps = [jsd_between_histograms(hists[a], hists[b]) for a in range(len(hists)) for b in range(a + 1, len(hists))]
            scores[j] = float(np.mean(ps)) if ps else 0.0
    return scores


def make_images(X, order, size):
    positions = spiral_order(size)
    imgs = np.zeros((X.shape[0], 1, size, size), dtype=np.float32)
    for i in range(X.shape[0]):
        for (r, c), j in zip(positions, order):
            if j < X.shape[1]:
                imgs[i, 0, r, c] = X[i, j]
    return imgs


def ml_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVC": SVC(kernel="linear", probability=False, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def train_torch(model, Xtr, ytr, Xte, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
    yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1) if binary else torch.tensor(ytr, dtype=torch.long)
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
    return torch.sigmoid(out).numpy().ravel() if binary else F.softmax(out, dim=1).numpy()


def load(task):
    labels = pd.read_csv(LABELS, sep="\t")
    cnv = pd.read_csv(D / "cnv_gene_case_matrix.tsv", sep="\t", index_col=0)
    if task == "PAM50_4class":
        labels = labels[labels["pam50_4class"].isin(["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"])]
    else:
        labels = labels[labels["os_event"].isin([0, 1]) & labels["os_time_days"].notna()]
    idx = [c for c in labels["case_id"] if c in cnv.index]
    X = np.nan_to_num(cnv.loc[idx].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    lab = labels.set_index("case_id").loc[idx]
    if task == "PAM50_4class":
        y = LabelEncoder().fit_transform(lab["pam50_4class"].values)
        return X, y, None, None
    return X, lab["os_event"].astype(int).values, lab["os_time_days"].astype(float).values, None


def run(task):
    X, y, ytime, _ = load(task)
    binary = task == "OS"
    k = PAM50_K if task == "PAM50_4class" else SURV_K
    size = PAM50_SIZE if task == "PAM50_4class" else SURV_SIZE
    names = list(ml_models().keys()) + ["MLP", "FullSizeCNN", "Dense"]
    acc = {n: [] for n in names}
    f1 = {n: [] for n in names}
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in cv.split(np.zeros(len(y)), y):
        pipe = build_adaptive_pipeline("CNV", task, k=k).fit(X[tr], y[tr])
        Xtr = pipe.transform(X[tr]); Xte = pipe.transform(X[te])
        order = np.argsort(jsd_scores(Xtr, y[tr]))[::-1]
        imgs_tr = make_images(Xtr[:, order], np.arange(Xtr.shape[1]), size)
        imgs_te = make_images(Xte[:, order], np.arange(Xte.shape[1]), size)
        for name, clf in ml_models().items():
            m = clf.fit(Xtr, y[tr])
            if binary:
                risk = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xte)
                acc[name].append(roc_auc_score(y[te], risk))
                f1[name].append(concordance_index(ytime[te], -risk, y[te]))
            else:
                acc[name].append(accuracy_score(y[te], m.predict(Xte)))
                f1[name].append(f1_score(y[te], m.predict(Xte), average="macro"))
        for nm, cls in [("MLP", MLP(Xtr.shape[1], 1 if binary else 4)),
                        ("FullSizeCNN", FullSizeCNN(size, 1 if binary else 4)),
                        ("Dense", Dense(Xtr.shape[1], 1 if binary else 4))]:
            inp = imgs_tr if nm == "FullSizeCNN" else Xtr
            inv = imgs_te if nm == "FullSizeCNN" else Xte
            prob = train_torch(cls, inp, y[tr], inv, binary)
            if binary:
                acc[nm].append(roc_auc_score(y[te], prob))
                f1[nm].append(concordance_index(ytime[te], -prob, y[te]))
            else:
                acc[nm].append(accuracy_score(y[te], prob.argmax(1)))
                f1[nm].append(f1_score(y[te], prob.argmax(1), average="macro"))
        print(f"[CNV {task}] fold done (n={len(y)})", flush=True)
    rows = []
    m1, m2 = ("roc_auc", "c_index") if binary else ("accuracy", "macro_f1")
    for nm in names:
        rows.append({"omics": "CNV", "task": "PAM50" if task == "PAM50_4class" else "Survival",
                     "model": nm, "metric": m1, "mean": round(float(np.mean(acc[nm])), 4), "std": round(float(np.std(acc[nm])), 4)})
        rows.append({"omics": "CNV", "task": "PAM50" if task == "PAM50_4class" else "Survival",
                     "model": nm, "metric": m2, "mean": round(float(np.mean(f1[nm])), 4), "std": round(float(np.std(f1[nm])), 4)})
    return rows


def main():
    rows = run("PAM50_4class") + run("OS")
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["omics", "task", "model", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print("done ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
