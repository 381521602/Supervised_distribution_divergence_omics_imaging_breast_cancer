# -*- coding: utf-8 -*-
"""Leak-free triple-omics fusion (fold-internal selection, scaling, JSD)."""
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
XENA = ROOT / "data" / "external" / "xena"
GH = ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_jsd import build_adaptive_pipeline, jsd_between_histograms
from run_multistream_cnn import spiral_order
from run_fullsize_multimodal_integration import Branch, train_multi, train_single_prob

LABELS = GH / "data" / "brca_labels_modeling_ready.tsv"
OUT = ROOT / "data" / "leakfree_triple_fusion_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
OMICS = ["mRNA", "CNV", "miRNA"]
RAW = {
    "mRNA": (XENA / "HiSeqV2", "mrna_sample_id"),
    "CNV": (XENA / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes", "cnv_gene_sample_id"),
    "miRNA": (XENA / "miRNA_HiSeq_gene", "mirna_sample_id"),
}
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


def load_omics(omics):
    path, col = RAW[omics]
    raw = pd.read_csv(path, sep="\t", index_col=0).T
    labels = pd.read_csv(LABELS, sep="\t").dropna(subset=[col])
    labels["prefix"] = labels[col].str[:15]
    labels = labels.drop_duplicates(subset=["prefix"], keep="first")
    return raw, labels.set_index("prefix")


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
    return torch.sigmoid(out).numpy().ravel() if binary else F.softmax(out, dim=1).numpy()


def prepare(task):
    raw = {}
    common = None
    for o in OMICS:
        r, lab = load_omics(o)
        if task == "PAM50_4class":
            m = lab["pam50_4class"].isin(["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"])
        else:
            m = lab["os_event"].isin([0, 1]) & lab["os_time_days"].notna()
        lab = lab[m]
        idx = [s for s in lab.index if s in r.index]
        X = np.nan_to_num(r.loc[idx].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        raw[o] = (r, lab, idx, X)
        cases = set(idx)
        common = cases if common is None else (common & cases)
    common = sorted(common)
    data = {}
    for o in OMICS:
        r, lab, _, _ = raw[o]
        idx = [s for s in common if s in r.index]
        X = np.nan_to_num(r.loc[idx].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if task == "PAM50_4class":
            y = LabelEncoder().fit_transform(lab.loc[idx, "pam50_4class"].values)
        else:
            y = lab.loc[idx, "os_event"].astype(int).values
        data[o] = (X, y)
    return data, common


def run_task(task):
    data, common = prepare(task)
    y0 = data[OMICS[0]][1]
    n_class = len(np.unique(y0)) if task == "PAM50_4class" else 1
    binary = task == "OS"
    metric = "roc_auc" if binary else "accuracy"
    K = PAM50_K if task == "PAM50_4class" else SURV_K
    SZ = PAM50_SIZE if task == "PAM50_4class" else SURV_SIZE
    schemes = {"Concat_LR": [], "Concat_MLP": [], "Concat_SVC": [], "CNN_LateAvg": [], "CNN_Gated": []}
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in cv.split(np.zeros(len(y0)), y0):
        Xtr_cat, Xte_cat, img_tr, img_te, sizes = [], [], {}, {}, {}
        for o in OMICS:
            X, y = data[o]
            pipe = build_adaptive_pipeline(o, task, k=K[o]).fit(X[tr], y[tr])
            Xt = pipe.transform(X[tr])
            Xv = pipe.transform(X[te])
            Xtr_cat.append(Xt)
            Xte_cat.append(Xv)
            order = np.argsort(jsd_scores(Xt, y[tr]))[::-1]
            img_tr[o] = make_images(Xt[:, order], np.arange(Xt.shape[1]), SZ[o])
            img_te[o] = make_images(Xv[:, order], np.arange(Xv.shape[1]), SZ[o])
            sizes[o] = SZ[o]
        Xtr_cat = np.hstack(Xtr_cat)
        Xte_cat = np.hstack(Xte_cat)
        ytr, yte = y0[tr], y0[te]
        pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
        pipe.fit(Xtr_cat, ytr)
        r = pipe.predict_proba(Xte_cat)[:, 1] if binary else pipe.predict(Xte_cat)
        schemes["Concat_LR"].append(roc_auc_score(yte, r) if binary else accuracy_score(yte, r))
        pipe = Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="linear", probability=False, random_state=RANDOM_STATE))])
        pipe.fit(Xtr_cat, ytr)
        r = pipe.decision_function(Xte_cat) if binary else pipe.predict(Xte_cat)
        schemes["Concat_SVC"].append(roc_auc_score(yte, r) if binary else accuracy_score(yte, r))
        prob = train_mlp(Xtr_cat, ytr, Xte_cat, binary)
        schemes["Concat_MLP"].append(roc_auc_score(yte, prob) if binary else accuracy_score(yte, prob.argmax(1)))
        probs = [train_single_prob(img_tr[o], ytr, img_te[o], sizes[o], binary) for o in OMICS]
        late = sum(probs) / len(probs)
        schemes["CNN_LateAvg"].append(roc_auc_score(yte, late) if binary else accuracy_score(yte, late.argmax(1)))
        p = train_multi(TripleGatedNet(sizes, 1 if binary else n_class), img_tr, ytr, img_te, binary)
        schemes["CNN_Gated"].append(roc_auc_score(yte, p) if binary else accuracy_score(yte, p.argmax(1)))
        print(f"[{task}] fold done (n={len(tr)})", flush=True)
    rows = []
    for s, vals in schemes.items():
        rows.append({"scheme": s, "task": "PAM50" if task == "PAM50_4class" else "Survival",
                     "metric": metric, "mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)})
    return rows


def main():
    rows = run_task("PAM50_4class") + run_task("OS")
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scheme", "task", "metric", "mean", "std"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print("FUSION done ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
