# -*- coding: utf-8 -*-
"""Full-method intersection single-omics and pairwise fusion (leak-free)."""
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

ROOT = Path(r"E:\组学数据图像化")
D = ROOT / "data"
XENA = D / "external" / "xena"
GH = ROOT / "github_release"
sys.path.insert(0, str(GH / "scripts"))
from adaptive_jsd import build_adaptive_pipeline, jsd_between_histograms
from run_multistream_cnn import spiral_order
from run_fullsize_multimodal_integration import Branch, train_multi, train_single_prob

LABELS = GH / "data" / "brca_labels_modeling_ready.tsv"
OUT_S = ROOT / "data" / "leakfree_intersection_single_full.tsv"
OUT_P = ROOT / "data" / "leakfree_pairwise_fusion_full.tsv"
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


class PairGated(nn.Module):
    def __init__(self, sizes, out_dim):
        super().__init__()
        self.branches = nn.ModuleDict({o: Branch(sizes[o], out_dim) for o in sizes})
        self.gate = nn.Sequential(nn.Linear(out_dim * 2, out_dim * 2), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(out_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))
    def forward(self, x):
        f = torch.cat([self.branches[o](x[o]) for o in self.branches], dim=1)
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


def ml_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVC": SVC(kernel="linear", probability=False, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


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
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
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


def align(data, labels, task, subset):
    cases = None
    for o in subset:
        _, c = data[o]
        cases = set(c) if cases is None else (cases & set(c))
    cases = sorted(cases)
    y = (LabelEncoder().fit_transform(labels.loc[cases, "pam50_4class"].values) if task == "PAM50_4class"
         else labels.loc[cases, "os_event"].astype(int).values)
    yt = None if task == "PAM50_4class" else labels.loc[cases, "os_time_days"].astype(float).values
    X = {}
    for o in subset:
        mat, c = data[o]
        pos = {cc: i for i, cc in enumerate(c)}
        X[o] = mat[[pos[cc] for cc in cases]]
    return X, y, yt, cases


def fold_sel(X, y, task, o, tr):
    k = (PAM50_K if task == "PAM50_4class" else SURV_K)[o]
    return build_adaptive_pipeline(o, task, k=k).fit(X[o][tr], y[tr])


def single(task):
    data, labels = load_omics(task)
    X, y, yt, cases = align(data, labels, task, OMICS)
    binary = task == "OS"
    m1, m2 = ("roc_auc", "c_index") if binary else ("accuracy", "macro_f1")
    names = list(ml_models().keys()) + ["MLP", "FullSizeCNN", "Dense"] + (["CoxPH"] if binary else [])
    res = {n: [] for n in names}
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for o in OMICS:
        for tr, te in cv.split(np.zeros(len(y)), y):
            pipe = fold_sel(X, y, task, o, tr)
            Xtr = pipe.transform(X[o][tr]); Xte = pipe.transform(X[o][te])
            SZ = (PAM50_SIZE if task == "PAM50_4class" else SURV_SIZE)[o]
            order = np.argsort(jsd_scores(Xtr, y[tr]))[::-1]
            itr = make_images(Xtr[:, order], np.arange(Xtr.shape[1]), SZ)
            ite = make_images(Xte[:, order], np.arange(Xte.shape[1]), SZ)
            for name, clf in ml_models().items():
                m = clf.fit(Xtr, y[tr])
                r = m.predict_proba(Xte)[:, 1] if binary and hasattr(m, "predict_proba") else (m.decision_function(Xte) if binary else m.predict(Xte))
                res[name].append((roc_auc_score(y[te], r) if binary else accuracy_score(y[te], r)))
            p = train_mlp(Xtr, y[tr], Xte, binary)
            res["MLP"].append(roc_auc_score(y[te], p) if binary else accuracy_score(y[te], p.argmax(1)))
            p = train_single_prob(itr, y[tr], ite, SZ, binary)
            res["FullSizeCNN"].append(roc_auc_score(y[te], p) if binary else accuracy_score(y[te], p.argmax(1)))
            # Dense baseline
            torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
            dm = Dense(Xtr.shape[1], 1 if binary else 4)
            opt = torch.optim.Adam(dm.parameters(), lr=1e-3, weight_decay=1e-4)
            Xtt = torch.tensor(Xtr, dtype=torch.float32)
            crit = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
            ytt = torch.tensor(y[tr], dtype=torch.float32).view(-1, 1) if binary else torch.tensor(y[tr], dtype=torch.long)
            nn_ = ytt.shape[0]
            for _ in range(EPOCHS):
                perm = torch.randperm(nn_)
                for i in range(0, nn_, BATCH_SIZE):
                    idx = perm[i:i + BATCH_SIZE]
                    if idx.shape[0] < 2:
                        continue
                    opt.zero_grad(); loss = crit(dm(Xtt[idx]), ytt[idx]); loss.backward(); opt.step()
            dm.eval()
            with torch.no_grad():
                pd_ = dm(torch.tensor(Xte, dtype=torch.float32))
            pd_ = torch.sigmoid(pd_).numpy().ravel() if binary else F.softmax(pd_, dim=1).numpy()
            res["Dense"].append(roc_auc_score(y[te], pd_) if binary else accuracy_score(y[te], pd_.argmax(1)))
            if binary:
                vtop = np.argsort(np.var(Xtr, axis=0))[::-1][:100]
                dftr = pd.DataFrame(Xtr[:, vtop]); dftr["time"] = yt[tr]; dftr["event"] = y[tr]
                dfte = pd.DataFrame(Xte[:, vtop])
                try:
                    cph = CoxPHFitter(penalizer=0.1).fit(dftr, duration_col="time", event_col="event")
                    h = cph.predict_partial_hazard(dfte)
                except Exception:
                    h = np.zeros(Xte.shape[0])
                res["CoxPH"].append(roc_auc_score(y[te], h))
        print(f"[single {o} {task}] done", flush=True)
    rows = []
    for o in OMICS:
        for nm in names:
            rows.append({"omics": o, "task": "PAM50" if task == "PAM50_4class" else "Survival",
                         "model": nm, "metric": m1, "mean": round(float(np.mean(res[nm])), 4),
                         "std": round(float(np.std(res[nm])), 4), "n": len(cases)})
    return rows


def pair(task, pr):
    data, labels = load_omics(task)
    X, y, yt, cases = align(data, labels, task, pr)
    binary = task == "OS"
    metric = "roc_auc" if binary else "accuracy"
    res = {"Concat_LR": [], "Concat_SVC": [], "Concat_MLP": [], "CNN_LateAvg": []}
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in cv.split(np.zeros(len(y)), y):
        trs, tes, itr, ite, sz = [], [], {}, {}, {}
        for o in pr:
            pipe = fold_sel(X, y, task, o, tr)
            Xt = pipe.transform(X[o][tr]); Xv = pipe.transform(X[o][te])
            trs.append(Xt); tes.append(Xv)
            SZ = (PAM50_SIZE if task == "PAM50_4class" else SURV_SIZE)[o]
            order = np.argsort(jsd_scores(Xt, y[tr]))[::-1]
            itr[o] = make_images(Xt[:, order], np.arange(Xt.shape[1]), SZ)
            ite[o] = make_images(Xv[:, order], np.arange(Xv.shape[1]), SZ)
            sz[o] = SZ
        Xtr = np.hstack(trs); Xte = np.hstack(tes)
        clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(Xtr, y[tr])
        r = clf.predict_proba(Xte)[:, 1] if binary else clf.predict(Xte)
        res["Concat_LR"].append(roc_auc_score(y[te], r) if binary else accuracy_score(y[te], r))
        svc = SVC(kernel="linear", probability=False, random_state=RANDOM_STATE).fit(Xtr, y[tr])
        r = svc.decision_function(Xte) if binary else svc.predict(Xte)
        res["Concat_SVC"].append(roc_auc_score(y[te], r) if binary else accuracy_score(y[te], r))
        p = train_mlp(Xtr, y[tr], Xte, binary)
        res["Concat_MLP"].append(roc_auc_score(y[te], p) if binary else accuracy_score(y[te], p.argmax(1)))
        probs = [train_single_prob(itr[o], y[tr], ite[o], sz[o], binary) for o in pr]
        late = sum(probs) / len(probs)
        res["CNN_LateAvg"].append(roc_auc_score(y[te], late) if binary else accuracy_score(y[te], late.argmax(1)))
        print(f"[pair {pr} {task}] fold done", flush=True)
    rows = []
    for s, vals in res.items():
        rows.append({"pair": "+".join(pr), "task": "PAM50" if task == "PAM50_4class" else "Survival",
                     "scheme": s, "metric": metric, "mean": round(float(np.mean(vals)), 4),
                     "std": round(float(np.std(vals)), 4), "n": len(cases)})
    return rows


def main():
    srows = single("PAM50_4class") + single("OS")
    with OUT_S.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["omics", "task", "model", "metric", "mean", "std", "n"], delimiter="\t")
        w.writeheader(); w.writerows(srows)
    prows = []
    for pr in [["mRNA", "CNV"], ["mRNA", "miRNA"], ["CNV", "miRNA"]]:
        prows += pair("PAM50_4class", pr) + pair("OS", pr)
    with OUT_P.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "task", "scheme", "metric", "mean", "std", "n"], delimiter="\t")
        w.writeheader(); w.writerows(prows)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
