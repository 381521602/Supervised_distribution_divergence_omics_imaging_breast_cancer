# -*- coding: utf-8 -*-
"""Leak-free single-omics rerun.

For each omics and task, feature selection (adaptive_jsd), scaling, JSD
ordering, and image generation are performed INSIDE each cross-validation
training fold. Results are written incrementally.
"""
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
XENA = ROOT / "data" / "external" / "xena"
GH = ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_jsd import build_adaptive_pipeline, jsd_between_histograms  # noqa: E402
from run_multistream_cnn import spiral_order  # noqa: E402

LABELS = GH / "data" / "brca_labels_modeling_ready.tsv"
OUT = ROOT / "data" / "leakfree_single_omics_results.tsv"

RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64

RAW = {
    "mRNA": (XENA / "HiSeqV2", "mrna_sample_id"),
    "CNV": (XENA / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes", "cnv_gene_sample_id"),
    "miRNA": (XENA / "miRNA_HiSeq_gene", "mirna_sample_id"),
}
PAM50_K = {"mRNA": 400, "CNV": 50, "miRNA": 600}
SURV_K = {"mRNA": 200, "CNV": 150, "miRNA": 600}
PAM50_SIZE = {"mRNA": 20, "CNV": 8, "miRNA": 25}
SURV_SIZE = {"mRNA": 15, "CNV": 13, "miRNA": 25}


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
            ps = [jsd_between_histograms(hists[a], hists[b])
                  for a in range(len(hists)) for b in range(a + 1, len(hists))]
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


class MLP(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, dout))

    def forward(self, x):
        return self.net(x)


class FullSizeCNN(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, out_dim))

    def forward(self, x):
        return self.head(F.relu(self.conv(x)))


class Dense(nn.Module):
    def __init__(self, din, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, x):
        return self.net(x)


def train_torch(model, Xtr, ytr, Xte, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    if binary:
        crit = nn.BCEWithLogitsLoss()
        yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    else:
        crit = nn.CrossEntropyLoss()
        yt = torch.tensor(ytr, dtype=torch.long)
    n = yt.shape[0]
    model.train()
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


def ml_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVC": SVC(kernel="linear", probability=False, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def load_aligned(omics):
    path, sample_col = RAW[omics]
    raw = pd.read_csv(path, sep="\t", index_col=0).T
    labels = pd.read_csv(LABELS, sep="\t")
    labels = labels.dropna(subset=[sample_col])
    labels["prefix"] = labels[sample_col].str[:15]
    labels = labels.drop_duplicates(subset=["prefix"], keep="first")
    idx = [s for s in labels["prefix"] if s in raw.index]
    X = raw.loc[idx].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, labels.set_index("prefix").loc[idx].reset_index(drop=True)


def eval_pam50(omics):
    X, lab = load_aligned(omics)
    m = lab["pam50_4class"].isin(["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"])
    X = X[m.to_numpy()]
    y = LabelEncoder().fit_transform(lab.loc[m, "pam50_4class"].values)
    k = PAM50_K[omics]
    size = PAM50_SIZE[omics]
    names = list(ml_models().keys()) + ["MLP", "FullSizeCNN", "Dense"]
    acc = {n: [] for n in names}
    f1 = {n: [] for n in names}
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in cv.split(np.zeros(len(y)), y):
        pipe = build_adaptive_pipeline(omics, "PAM50_4class", k=k).fit(X[tr], y[tr])
        Xtr = pipe.transform(X[tr])
        Xte = pipe.transform(X[te])
        order = np.argsort(jsd_scores(Xtr, y[tr]))[::-1]
        imgs_tr = make_images(Xtr[:, order], np.arange(Xtr.shape[1]), size)
        imgs_te = make_images(Xte[:, order], np.arange(Xte.shape[1]), size)
        for name, clf in ml_models().items():
            mdl = clf.fit(Xtr, y[tr])
            pred = mdl.predict(Xte)
            acc[name].append(accuracy_score(y[te], pred))
            f1[name].append(f1_score(y[te], pred, average="macro"))
        p = train_torch(MLP(Xtr.shape[1], 4), Xtr, y[tr], Xte, binary=False)
        acc["MLP"].append(accuracy_score(y[te], p.argmax(1)))
        f1["MLP"].append(f1_score(y[te], p.argmax(1), average="macro"))
        p = train_torch(FullSizeCNN(size, 4), imgs_tr, y[tr], imgs_te, binary=False)
        acc["FullSizeCNN"].append(accuracy_score(y[te], p.argmax(1)))
        f1["FullSizeCNN"].append(f1_score(y[te], p.argmax(1), average="macro"))
        p = train_torch(Dense(Xtr.shape[1], 4), Xtr, y[tr], Xte, binary=False)
        acc["Dense"].append(accuracy_score(y[te], p.argmax(1)))
        f1["Dense"].append(f1_score(y[te], p.argmax(1), average="macro"))
        print(f"[PAM50 {omics}] fold done (k={Xtr.shape[1]})", flush=True)
    rows = []
    for nm in names:
        rows.append({"omics": omics, "task": "PAM50", "model": nm, "metric": "accuracy",
                     "mean": round(float(np.mean(acc[nm])), 4), "std": round(float(np.std(acc[nm])), 4)})
        rows.append({"omics": omics, "task": "PAM50", "model": nm, "metric": "macro_f1",
                     "mean": round(float(np.mean(f1[nm])), 4), "std": round(float(np.std(f1[nm])), 4)})
    return rows


def eval_survival(omics):
    X, lab = load_aligned(omics)
    m = lab["os_event"].isin([0, 1]) & lab["os_time_days"].notna()
    X = X[m.to_numpy()]
    y_event = lab.loc[m, "os_event"].astype(int).values
    y_time = lab.loc[m, "os_time_days"].astype(float).values
    k = SURV_K[omics]
    size = SURV_SIZE[omics]
    names = list(ml_models().keys()) + ["MLP", "FullSizeCNN", "Dense", "CoxPH"]
    auc = {n: [] for n in names}
    ci = {n: [] for n in names}
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        pipe = build_adaptive_pipeline(omics, "OS", k=k).fit(X[tr], y_event[tr])
        Xtr = pipe.transform(X[tr])
        Xte = pipe.transform(X[te])
        order = np.argsort(jsd_scores(Xtr, y_event[tr]))[::-1]
        imgs_tr = make_images(Xtr[:, order], np.arange(Xtr.shape[1]), size)
        imgs_te = make_images(Xte[:, order], np.arange(Xte.shape[1]), size)
        for name, clf in ml_models().items():
            mdl = clf.fit(Xtr, y_event[tr])
            risk = mdl.predict_proba(Xte)[:, 1] if hasattr(mdl, "predict_proba") else mdl.decision_function(Xte)
            auc[name].append(roc_auc_score(y_event[te], risk))
            ci[name].append(concordance_index(y_time[te], -risk, y_event[te]))
        risk = train_torch(MLP(Xtr.shape[1], 1), Xtr, y_event[tr], Xte, binary=True)
        auc["MLP"].append(roc_auc_score(y_event[te], risk))
        ci["MLP"].append(concordance_index(y_time[te], -risk, y_event[te]))
        risk = train_torch(FullSizeCNN(size, 1), imgs_tr, y_event[tr], imgs_te, binary=True)
        auc["FullSizeCNN"].append(roc_auc_score(y_event[te], risk))
        ci["FullSizeCNN"].append(concordance_index(y_time[te], -risk, y_event[te]))
        risk = train_torch(Dense(Xtr.shape[1], 1), Xtr, y_event[tr], Xte, binary=True)
        auc["Dense"].append(roc_auc_score(y_event[te], risk))
        ci["Dense"].append(concordance_index(y_time[te], -risk, y_event[te]))
        vtop = np.argsort(np.var(Xtr, axis=0))[::-1][:100]
        df_tr = pd.DataFrame(Xtr[:, vtop]); df_tr["time"] = y_time[tr]; df_tr["event"] = y_event[tr]
        df_te = pd.DataFrame(Xte[:, vtop])
        try:
            cph = CoxPHFitter(penalizer=0.1).fit(df_tr, duration_col="time", event_col="event")
            hazard = cph.predict_partial_hazard(df_te)
        except Exception:
            hazard = np.zeros(Xte.shape[0])
        auc["CoxPH"].append(roc_auc_score(y_event[te], hazard))
        ci["CoxPH"].append(concordance_index(y_time[te], -hazard, y_event[te]))
        print(f"[Survival {omics}] fold done (k={Xtr.shape[1]})", flush=True)
    rows = []
    for nm in names:
        rows.append({"omics": omics, "task": "Survival", "model": nm, "metric": "roc_auc",
                     "mean": round(float(np.mean(auc[nm])), 4), "std": round(float(np.std(auc[nm])), 4)})
        rows.append({"omics": omics, "task": "Survival", "model": nm, "metric": "c_index",
                     "mean": round(float(np.mean(ci[nm])), 4), "std": round(float(np.std(ci[nm])), 4)})
    return rows


def write_rows(rows, append=True):
    cols = ["omics", "task", "model", "metric", "mean", "std"]
    mode = "a" if (append and OUT.exists()) else "w"
    with OUT.open(mode, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        if mode == "w":
            w.writeheader()
        w.writerows(rows)


def main():
    quick = "--quick" in sys.argv
    if quick:
        write_rows(eval_pam50("mRNA"), append=False)
        print("QUICK done: mRNA PAM50")
        return 0
    for omics in ["mRNA", "CNV", "miRNA"]:
        write_rows(eval_pam50(omics))
        write_rows(eval_survival(omics))
    print("FULL done ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
