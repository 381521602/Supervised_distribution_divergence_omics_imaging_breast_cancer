# -*- coding: utf-8 -*-
"""Leak-free Transformer and DeepSurv rerun on the triple-omics intersection."""
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
XENA = D / "external" / "xena"
GH = ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_jsd import build_adaptive_pipeline

LABELS = GH / "data" / "brca_labels_modeling_ready.tsv"
OUT = ROOT / "data" / "leakfree_advanced_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
OMICS = ["mRNA", "CNV", "miRNA"]
PAM50_K = {"mRNA": 400, "CNV": 50, "miRNA": 600}
SURV_K = {"mRNA": 200, "CNV": 150, "miRNA": 600}


class TransformerFusion(nn.Module):
    def __init__(self, dims, d_model=32, nhead=4, layers=2, out_dim=1):
        super().__init__()
        self.proj = nn.ModuleDict({o: nn.Linear(dims[o], d_model) for o in OMICS})
        el = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, xs):
        tokens = torch.stack([self.proj[o](xs[o]) for o in OMICS], dim=1)
        z = self.encoder(tokens).mean(dim=1)
        return self.head(z)


class DeepSurv(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.proj = nn.ModuleDict({o: nn.Linear(dims[o], 32) for o in OMICS})
        self.net = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, xs):
        f = torch.cat([self.proj[o](xs[o]) for o in OMICS], dim=1)
        return self.net(f)


def cox_loss(risk, time, event):
    n = risk.shape[0]
    loss = torch.zeros((), dtype=risk.dtype)
    for i in range(n):
        if event[i] == 0:
            continue
        at_risk = (time >= time[i]).float()
        logsum = torch.logsumexp(risk + torch.log(at_risk + 1e-12), dim=0)
        loss = loss + logsum - risk[i]
    return loss / max(1, int(event.sum().item()))


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
    ytime = None if task == "PAM50_4class" else lab.loc[cases, "os_time_days"].astype(float).values
    X = {}
    for o in OMICS:
        mat, c = data[o]
        pos = {cc: i for i, cc in enumerate(c)}
        X[o] = mat[[pos[cc] for cc in cases]]
    return X, y, ytime, cases


def fold_prep(X, y, task, tr, te):
    K = PAM50_K if task == "PAM50_4class" else SURV_K
    Xtr, Xte, dims = {}, {}, {}
    for o in OMICS:
        pipe = build_adaptive_pipeline(o, task, k=K[o]).fit(X[o][tr], y[tr])
        Xtr[o] = pipe.transform(X[o][tr])
        Xte[o] = pipe.transform(X[o][te])
        dims[o] = Xtr[o].shape[1]
    return Xtr, Xte, dims


def train_transformer(Xtr, ytr, Xte, dims, binary):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = TransformerFusion(dims, out_dim=1 if binary else 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = {o: torch.tensor(v, dtype=torch.float32) for o, v in Xtr.items()}
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
            loss = crit(model({o: v[idx] for o, v in Xt.items()}), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model({o: torch.tensor(v, dtype=torch.float32) for o, v in Xte.items()})
    return torch.sigmoid(out).numpy().ravel() if binary else F.softmax(out, dim=1).numpy()


def train_deepsurv(Xtr, ytr_time, ytr_event, Xte):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    dims = {o: Xtr[o].shape[1] for o in OMICS}
    model = DeepSurv(dims)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = {o: torch.tensor(v, dtype=torch.float32) for o, v in Xtr.items()}
    time = torch.tensor(ytr_time, dtype=torch.float32)
    event = torch.tensor(ytr_event, dtype=torch.float32)
    n = event.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if idx.shape[0] < 2 or event[idx].sum().item() == 0:
                continue
            opt.zero_grad()
            risk = model({o: v[idx] for o, v in Xt.items()}).view(-1)
            loss = cox_loss(risk, time[idx], event[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model({o: torch.tensor(v, dtype=torch.float32) for o, v in Xte.items()}).numpy().ravel()


def run(task):
    X, y, ytime, cases = load_omics(task)
    binary = task == "OS"
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    acc, f1, auc, ci = [], [], [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        Xtr, Xte, dims = fold_prep(X, y, task, tr, te)
        if binary:
            p = train_transformer(Xtr, y[tr], Xte, dims, True)
            auc.append(roc_auc_score(y[te], p))
            ci.append(concordance_index(ytime[te], -p, y[te]))
            p = train_deepsurv(Xtr, ytime[tr], y[tr], Xte)
            auc.append(roc_auc_score(y[te], p))
            ci.append(concordance_index(ytime[te], -p, y[te]))
        else:
            p = train_transformer(Xtr, y[tr], Xte, dims, False)
            acc.append(accuracy_score(y[te], p.argmax(1)))
            f1.append(f1_score(y[te], p.argmax(1), average="macro"))
        print(f"[{task}] fold done (n={len(cases)})", flush=True)
    rows = []
    tname = "PAM50" if task == "PAM50_4class" else "Survival"
    if binary:
        # transformer (first 5) and deepsurv (last 5)
        for name, va, vc in [("Transformer", auc[:K_FOLDS], ci[:K_FOLDS]), ("DeepSurv", auc[K_FOLDS:], ci[K_FOLDS:])]:
            rows.append({"method": name, "task": tname, "metric": "roc_auc",
                         "mean": round(float(np.mean(va)), 4), "std": round(float(np.std(va)), 4), "n": len(cases)})
            rows.append({"method": name, "task": tname, "metric": "c_index",
                         "mean": round(float(np.mean(vc)), 4), "std": round(float(np.std(vc)), 4), "n": len(cases)})
    else:
        rows.append({"method": "Transformer", "task": tname, "metric": "accuracy",
                     "mean": round(float(np.mean(acc)), 4), "std": round(float(np.std(acc)), 4), "n": len(cases)})
        rows.append({"method": "Transformer", "task": tname, "metric": "macro_f1",
                     "mean": round(float(np.mean(f1)), 4), "std": round(float(np.std(f1)), 4), "n": len(cases)})
    return rows


def main():
    rows = run("PAM50_4class") + run("OS")
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "task", "metric", "mean", "std", "n"], delimiter="\t")
        w.writeheader(); w.writerows(rows)
    print("done ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
