# -*- coding: utf-8 -*-
"""Leak-free intersection single-omics baselines and pairwise fusion."""
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(r"E:\组学数据图像化")
D = ROOT / "data"
XENA = D / "external" / "xena"
GH = ROOT / "github_release"
sys.path.insert(0, str(GH / "scripts"))
from adaptive_jsd import build_adaptive_pipeline, jsd_between_histograms
from run_multistream_cnn import spiral_order

LABELS = GH / "data" / "brca_labels_modeling_ready.tsv"
OUT_SINGLE = ROOT / "data" / "leakfree_intersection_single.tsv"
OUT_PAIR = ROOT / "data" / "leakfree_pairwise_fusion.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
OMICS = ["mRNA", "CNV", "miRNA"]
PAM50_K = {"mRNA": 400, "CNV": 50, "miRNA": 600}
SURV_K = {"mRNA": 200, "CNV": 150, "miRNA": 600}


class MLP(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, dout))
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


def train_mlp(Xtr, ytr, Xte, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = MLP(Xtr.shape[1], 1 if binary else 4)
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


def load_omics(task):
    labels = pd.read_csv(LABELS, sep="\t")
    if task == "PAM50_4class":
        labels = labels[labels["pam50_4class"].isin(["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"])]
    else:
        labels = labels[labels["os_event"].isin([0, 1]) & labels["os_time_days"].notna()]
    data = {}
    for o, (fname, col) in {
        "mRNA": (XENA / "HiSeqV2", "mrna_sample_id"),
        "miRNA": (XENA / "miRNA_HiSeq_gene", "mirna_sample_id"),
    }.items():
        raw = pd.read_csv(fname, sep="\t", index_col=0).T
        lab = labels.dropna(subset=[col]).drop_duplicates(subset=["case_id"]).copy()
        lab["prefix"] = lab[col].str[:15]
        idx = [s for s in lab["prefix"] if s in raw.index]
        lab2 = lab.set_index("prefix").loc[idx]
        X = np.nan_to_num(raw.loc[idx].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        data[o] = (X, lab2["case_id"].tolist())
    cnv = pd.read_csv(D / "cnv_gene_case_matrix.tsv", sep="\t", index_col=0)
    idx = [c for c in labels["case_id"] if c in cnv.index]
    X = np.nan_to_num(cnv.loc[idx].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    data["CNV"] = (X, idx)
    return data, labels.set_index("case_id")


def intersect(data, labels, task, omics_subset):
    cases = None
    for o in omics_subset:
        _, c = data[o]
        cases = set(c) if cases is None else (cases & set(c))
    cases = sorted(cases)
    y = (LabelEncoder().fit_transform(labels.loc[cases, "pam50_4class"].values) if task == "PAM50_4class"
         else labels.loc[cases, "os_event"].astype(int).values)
    X = {}
    for o in omics_subset:
        mat, c = data[o]
        pos = {cc: i for i, cc in enumerate(c)}
        X[o] = mat[[pos[cc] for cc in cases]]
    return X, y, cases


def fold_select(X, y, task, o):
    k = (PAM50_K if task == "PAM50_4class" else SURV_K)[o]
    pipe = build_adaptive_pipeline(o, task, k=k)
    return pipe


def run_pair(pair, task):
    data, labels = load_omics(task)
    X, y, cases = intersect(data, labels, task, pair)
    binary = task == "OS"
    metric = "roc_auc" if binary else "accuracy"
    res = {"Concat_LR": [], "Concat_MLP": []}
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in cv.split(np.zeros(len(y)), y):
        trs, tes = [], []
        for o in pair:
            pipe = fold_select(X[o], y, task, o).fit(X[o][tr], y[tr])
            trs.append(pipe.transform(X[o][tr]))
            tes.append(pipe.transform(X[o][te]))
        Xtr = np.hstack(trs); Xte = np.hstack(tes)
        clf = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
        clf.fit(Xtr, y[tr])
        r = clf.predict_proba(Xte)[:, 1] if binary else clf.predict(Xte)
        res["Concat_LR"].append(roc_auc_score(y[te], r) if binary else accuracy_score(y[te], r))
        prob = train_mlp(Xtr, y[tr], Xte, binary)
        res["Concat_MLP"].append(roc_auc_score(y[te], prob) if binary else accuracy_score(y[te], prob.argmax(1)))
        print(f"[pair {pair} {task}] fold done (n={len(cases)})", flush=True)
    rows = []
    for s, vals in res.items():
        rows.append({"pair": "+".join(pair), "task": "PAM50" if task == "PAM50_4class" else "Survival",
                     "scheme": s, "metric": metric,
                     "mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                     "n": len(cases)})
    return rows


def run_intersection_single(task):
    data, labels = load_omics(task)
    X, y, cases = intersect(data, labels, task, OMICS)
    binary = task == "OS"
    metric = "roc_auc" if binary else "accuracy"
    rows = []
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for o in OMICS:
        acc = []
        for tr, te in cv.split(np.zeros(len(y)), y):
            pipe = fold_select(X[o], y, task, o).fit(X[o][tr], y[tr])
            Xtr = pipe.transform(X[o][tr]); Xte = pipe.transform(X[o][te])
            clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(Xtr, y[tr])
            r = clf.predict_proba(Xte)[:, 1] if binary else clf.predict(Xte)
            acc.append(roc_auc_score(y[te], r) if binary else accuracy_score(y[te], r))
        rows.append({"omics": o, "task": "PAM50" if task == "PAM50_4class" else "Survival",
                     "model": "LogisticRegression", "metric": metric,
                     "mean": round(float(np.mean(acc)), 4), "std": round(float(np.std(acc)), 4),
                     "n": len(cases)})
        print(f"[single {o} {task}] done (n={len(cases)})", flush=True)
    return rows


def main():
    single = run_intersection_single("PAM50_4class") + run_intersection_single("OS")
    with OUT_SINGLE.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["omics", "task", "model", "metric", "mean", "std", "n"], delimiter="\t")
        w.writeheader(); w.writerows(single)
    pairs = []
    for pair in [["mRNA", "CNV"], ["mRNA", "miRNA"], ["CNV", "miRNA"]]:
        pairs += run_pair(pair, "PAM50_4class") + run_pair(pair, "OS")
    with OUT_PAIR.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "task", "scheme", "metric", "mean", "std", "n"], delimiter="\t")
        w.writeheader(); w.writerows(pairs)
    print("done ->", OUT_SINGLE, OUT_PAIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
