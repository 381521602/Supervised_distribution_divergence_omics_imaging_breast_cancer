# -*- coding: utf-8 -*-
"""Recompute paired t-tests / Wilcoxon between best fusion and best single-omics
using fold-level metrics from the same patients and the same CV folds."""
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
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
XENA = D / "external" / "xena"
GH = ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_jsd import build_adaptive_pipeline, jsd_between_histograms
from run_multistream_cnn import spiral_order
from run_fullsize_multimodal_integration import train_single_prob, train_multi, Branch

LABELS = GH / "data" / "brca_labels_modeling_ready.tsv"
OUT = ROOT / "data" / "leakfree_paired_tests.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
OMICS = ["mRNA", "CNV", "miRNA"]
PAM50_K = {"mRNA": 400, "CNV": 50, "miRNA": 600}
SURV_K = {"mRNA": 200, "CNV": 150, "miRNA": 600}
PAM50_SIZE = {"mRNA": 20, "CNV": 8, "miRNA": 25}
SURV_SIZE = {"mRNA": 15, "CNV": 13, "miRNA": 25}


class MLP(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, dout))
    def forward(self, x):
        return self.net(x)


class TripleGated(nn.Module):
    def __init__(self, sizes, out_dim):
        super().__init__()
        self.branches = nn.ModuleDict({o: Branch(sizes[o], out_dim) for o in OMICS})
        self.gate = nn.Sequential(nn.Linear(out_dim * 3, out_dim * 3), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(out_dim * 3, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))
    def forward(self, x):
        f = torch.cat([self.branches[o](x[o]) for o in OMICS], dim=1)
        return self.head(f * self.gate(f))


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
    cases = None
    for o in OMICS:
        _, c = data[o]
        cases = set(c) if cases is None else (cases & set(c))
    cases = sorted(cases)
    lab = labels.set_index("case_id")
    y = (LabelEncoder().fit_transform(lab.loc[cases, "pam50_4class"].values) if task == "PAM50_4class"
         else lab.loc[cases, "os_event"].astype(int).values)
    yt = None if task == "PAM50_4class" else lab.loc[cases, "os_time_days"].astype(float).values
    X = {}
    for o in OMICS:
        mat, c = data[o]
        pos = {cc: i for i, cc in enumerate(c)}
        X[o] = mat[[pos[cc] for cc in cases]]
    return X, y, yt, cases


def train_mlp(Xtr, ytr, Xte, binary):
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
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
            opt.zero_grad(); loss = crit(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte, dtype=torch.float32))
    return torch.sigmoid(out).numpy().ravel() if binary else F.softmax(out, dim=1).numpy()


def run(task):
    X, y, yt, cases = load_omics(task)
    binary = task == "OS"
    K = PAM50_K if task == "PAM50_4class" else SURV_K
    SZ = PAM50_SIZE if task == "PAM50_4class" else SURV_SIZE
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for tr, te in cv.split(np.zeros(len(y)), y):
        # single-omics mRNA RandomForest (best intersection single-omics)
        pipe = build_adaptive_pipeline("mRNA", task, k=K["mRNA"]).fit(X["mRNA"][tr], y[tr])
        Xtr = pipe.transform(X["mRNA"][tr]); Xte = pipe.transform(X["mRNA"][te])
        if binary:
            m = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr, y[tr])
            risk = m.predict_proba(Xte)[:, 1]
            single = roc_auc_score(y[te], risk)
        else:
            m = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr, y[tr])
            single = accuracy_score(y[te], m.predict(Xte))
        # fusion: fold-internal selection for all omics + Concat MLP and CNN late-average
        trs, tes, itr, ite, sz = [], [], {}, {}, {}
        for o in OMICS:
            pipe = build_adaptive_pipeline(o, task, k=K[o]).fit(X[o][tr], y[tr])
            Xt = pipe.transform(X[o][tr]); Xv = pipe.transform(X[o][te])
            trs.append(Xt); tes.append(Xv)
            S = SZ[o]
            order = np.argsort(jsd_scores(Xt, y[tr]))[::-1]
            itr[o] = make_images(Xt[:, order], np.arange(Xt.shape[1]), S)
            ite[o] = make_images(Xv[:, order], np.arange(Xv.shape[1]), S)
            sz[o] = S
        Xtr_c = np.hstack(trs); Xte_c = np.hstack(tes)
        p = train_mlp(Xtr_c, y[tr], Xte_c, binary)
        mlp_fusion = roc_auc_score(y[te], p) if binary else accuracy_score(y[te], p.argmax(1))
        probs = [train_single_prob(itr[o], y[tr], ite[o], sz[o], binary) for o in OMICS]
        late = sum(probs) / len(probs)
        cnn_fusion = roc_auc_score(y[te], late) if binary else accuracy_score(y[te], late.argmax(1))
        folds.append({"fold": len(folds), "single": single, "mlp_fusion": mlp_fusion, "cnn_fusion": cnn_fusion})
        print(f"[{task}] fold done", flush=True)
    rows = []
    for name, col in [("Concat_MLP", "mlp_fusion"), ("CNN_LateAvg", "cnn_fusion")]:
        s = [f["single"] for f in folds]
        f = [f[col] for f in folds]
        t = stats.ttest_rel(f, s)
        w = stats.wilcoxon(f, s)
        rows.append({"task": "PAM50" if task == "PAM50_4class" else "Survival", "fusion": name,
                     "fusion_mean": round(float(np.mean(f)), 4), "single_mean": round(float(np.mean(s)), 4),
                     "mean_diff": round(float(np.mean(np.array(f) - np.array(s))), 4),
                     "ttest_p": round(float(t.pvalue), 4), "wilcoxon_p": round(float(w.pvalue), 4),
                     "n": len(cases)})
    return rows


def main():
    rows = run("PAM50_4class") + run("OS")
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "fusion", "fusion_mean", "single_mean", "mean_diff", "ttest_p", "wilcoxon_p", "n"], delimiter="\t")
        w.writeheader(); w.writerows(rows)
    print("done ->", OUT)
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
