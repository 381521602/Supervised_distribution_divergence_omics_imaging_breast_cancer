# -*- coding: utf-8 -*-
"""Rerun CNV single-omics and triple-omics fusion with the corrected GDC CNV
matrix (case_id x genes), aligning all omics by case_id."""
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
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
XENA = D / "external" / "xena"
GH = ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_jsd import build_adaptive_pipeline, jsd_between_histograms
from run_multistream_cnn import spiral_order
from run_fullsize_multimodal_integration import Branch, train_multi, train_single_prob

LABELS = GH / "data" / "brca_labels_modeling_ready.tsv"
OUT = ROOT / "data" / "leakfree_cnv_fusion_rerun.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
OMICS = ["mRNA", "CNV", "miRNA"]
PAM50_K = {"mRNA": 400, "CNV": 50, "miRNA": 600}
SURV_K = {"mRNA": 200, "CNV": 150, "miRNA": 600}
PAM50_SIZE = {"mRNA": 20, "CNV": 8, "miRNA": 25}
SURV_SIZE = {"mRNA": 15, "CNV": 13, "miRNA": 25}


class TripleGatedNet(nn.Module):
    def __init__(self, sizes, out_dim):
        super().__init__()
        self.branches = nn.ModuleDict({o: Branch(sizes[o], out_dim) for o in OMICS})
        self.gate = nn.Sequential(nn.Linear(out_dim * 3, out_dim * 3), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(out_dim * 3, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))
    def forward(self, x):
        f = torch.cat([self.branches[o](x[o]) for o in OMICS], dim=1)
        return self.head(f * self.gate(f))


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


def make_images(X, order, size):
    positions = spiral_order(size)
    imgs = np.zeros((X.shape[0], 1, size, size), dtype=np.float32)
    for i in range(X.shape[0]):
        for (r, c), j in zip(positions, order):
            if j < X.shape[1]:
                imgs[i, 0, r, c] = X[i, j]
    return imgs


def load_omics_by_case(task):
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
        lab = labels.dropna(subset=[col]).copy()
        lab["prefix"] = lab[col].str[:15]
        lab = lab.drop_duplicates(subset=["case_id"])
        idx = [s for s in lab["prefix"] if s in raw.index]
        lab2 = lab.set_index("prefix").loc[idx]
        X = np.nan_to_num(raw.loc[idx].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        data[o] = (X, lab2["case_id"].tolist())
    cnv = pd.read_csv(D / "cnv_gene_case_matrix.tsv", sep="\t", index_col=0)
    lab = labels.dropna(subset=["case_id"])
    idx = [c for c in lab["case_id"] if c in cnv.index]
    X = np.nan_to_num(cnv.loc[idx].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    data["CNV"] = (X, idx)
    return data, labels


def align(data, labels, task):
    cases = None
    for o in OMICS:
        _, c = data[o]
        cases = set(c) if cases is None else (cases & set(c))
    cases = sorted(cases)
    lab = labels.set_index("case_id")
    y = (LabelEncoder().fit_transform(lab.loc[cases, "pam50_4class"].values) if task == "PAM50_4class"
         else lab.loc[cases, "os_event"].astype(int).values)
    X = {}
    for o in OMICS:
        mat, c = data[o]
        pos = {cc: i for i, cc in enumerate(c)}
        idx = [pos[cc] for cc in cases]
        X[o] = mat[idx]
    return X, y, cases


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


def run(task):
    data, labels = load_omics_by_case(task)
    X, y, cases = align(data, labels, task)
    binary = task == "OS"
    n_class = len(np.unique(y)) if not binary else 1
    K = PAM50_K if task == "PAM50_4class" else SURV_K
    SZ = PAM50_SIZE if task == "PAM50_4class" else SURV_SIZE
    metric = "roc_auc" if binary else "accuracy"
    schemes = {"Concat_LR": [], "Concat_MLP": [], "CNN_LateAvg": [], "CNN_Gated": []}
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in cv.split(np.zeros(len(y)), y):
        Xtr_cat, Xte_cat, img_tr, img_te, sizes = [], [], {}, {}, {}
        for o in OMICS:
            pipe = build_adaptive_pipeline(o, task, k=K[o]).fit(X[o][tr], y[tr])
            Xt = pipe.transform(X[o][tr])
            Xv = pipe.transform(X[o][te])
            Xtr_cat.append(Xt); Xte_cat.append(Xv)
            order = np.argsort(jsd_scores(Xt, y[tr]))[::-1]
            img_tr[o] = make_images(Xt[:, order], np.arange(Xt.shape[1]), SZ[o])
            img_te[o] = make_images(Xv[:, order], np.arange(Xv.shape[1]), SZ[o])
            sizes[o] = SZ[o]
        Xtr_cat = np.hstack(Xtr_cat); Xte_cat = np.hstack(Xte_cat)
        pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
        pipe.fit(Xtr_cat, y[tr])
        r = pipe.predict_proba(Xte_cat)[:, 1] if binary else pipe.predict(Xte_cat)
        schemes["Concat_LR"].append(roc_auc_score(y[te], r) if binary else accuracy_score(y[te], r))
        prob = train_mlp(Xtr_cat, y[tr], Xte_cat, binary)
        schemes["Concat_MLP"].append(roc_auc_score(y[te], prob) if binary else accuracy_score(y[te], prob.argmax(1)))
        probs = [train_single_prob(img_tr[o], y[tr], img_te[o], sizes[o], binary) for o in OMICS]
        late = sum(probs) / len(probs)
        schemes["CNN_LateAvg"].append(roc_auc_score(y[te], late) if binary else accuracy_score(y[te], late.argmax(1)))
        p = train_multi(TripleGatedNet(sizes, 1 if binary else n_class), img_tr, y[tr], img_te, binary)
        schemes["CNN_Gated"].append(roc_auc_score(y[te], p) if binary else accuracy_score(y[te], p.argmax(1)))
        print(f"[{task}] fold done (n={len(cases)})", flush=True)
    rows = []
    for s, vals in schemes.items():
        rows.append({"scheme": s, "task": "PAM50" if task == "PAM50_4class" else "Survival",
                     "metric": metric, "mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                     "n_samples": len(cases)})
    return rows


def main():
    rows = run("PAM50_4class") + run("OS")
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scheme", "task", "metric", "mean", "std", "n_samples"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print("done ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
