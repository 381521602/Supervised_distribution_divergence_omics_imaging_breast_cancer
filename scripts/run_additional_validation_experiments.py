from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import jsd_scores, spiral_order  # noqa: E402

DATA = ROOT / "data"
HISEQ = DATA / "external/xena/HiSeqV2"
PAM_FINAL = DATA / "final_datasets/PAM50/mRNA_PAM50_final.tsv"
SUR_FINAL = DATA / "final_datasets/Survival/mRNA_Survival_final.tsv"
SUR_IMG = DATA / "images/Survival/mRNA/images.npy"
OUT_NESTED = DATA / "nested_cv_pam50_mrna_results.tsv"
OUT_COX = DATA / "fullsize_cnn_cox_survival_results.tsv"
OUT_EXCL = DATA / "strict_pam50_exclusion_results.tsv"

RANDOM_STATE = 42
DEVICE = torch.device("cpu")
EPOCHS = 30
BATCH_SIZE = 64

PAM50_GENES = {
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "NUF2", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR",
    "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7",
    "KIF2C", "NDC80", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
}


def make_selector(k: int):
    """Low-variance filter + L1/LASSO embedded feature selection."""
    return Pipeline(
        [
            ("var", VarianceThreshold(threshold=0.0)),
            (
            "l1",
            SelectFromModel(
                LinearSVC(
                    penalty="l1",
                    dual=False,
                    C=0.1,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
                max_features=k,
                threshold=-np.inf,
            ),
            ),
        ]
    )


class FullSizeCNN(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.head(F.relu(self.conv(x)))


class DenseBaseline(nn.Module):
    def __init__(self, din, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FullSizeCNNCox(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(F.relu(self.conv(x)))


def make_images(X: np.ndarray, order: np.ndarray, size: int) -> np.ndarray:
    positions = spiral_order(size)
    imgs = np.zeros((X.shape[0], 1, size, size), dtype=np.float32)
    for i in range(X.shape[0]):
        for (r, c), feat_idx in zip(positions, order):
            if feat_idx < X.shape[1]:
                imgs[i, 0, r, c] = X[i, feat_idx]
    return imgs


def train_torch_bce(model, Xtr, ytr, Xte, binary, size=None):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = model.to(DEVICE)
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
    if binary:
        return torch.sigmoid(out).numpy().ravel()
    return F.softmax(out, dim=1).numpy()


def nested_pam50_mrna():
    expr = pd.read_csv(HISEQ, sep="\t", index_col=0).T
    labels = pd.read_csv(DATA / "brca_labels_modeling_ready.tsv", sep="\t")
    labels = labels[labels["pam50_4class"].isin(["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"])]
    labels = labels.dropna(subset=["mrna_sample_id"])
    labels["prefix"] = labels["mrna_sample_id"].str[:15]
    idx = [s for s in labels["prefix"] if s in expr.index]
    X = expr.loc[idx].to_numpy(dtype=float)
    y_raw = labels.set_index("prefix").loc[idx, "pam50_4class"].values
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)

    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    candidate_ks = [200, 400]
    metrics = {"LogisticRegression": {"acc": [], "f1": []}, "FullSizeCNN": {"acc": [], "f1": []}, "Dense": {"acc": [], "f1": []}}
    selected_counts = []

    for tr, te in outer.split(np.zeros(len(y)), y):
        best_k, best_score = candidate_ks[0], -1.0
        for k in candidate_ks:
            scores = []
            for itr, ite in inner.split(np.zeros(len(y[tr])), y[tr]):
                sel = make_selector(k).fit(X[tr][itr], y[tr][itr])
                raw_tr = sel.transform(X[tr][itr])
                raw_te = sel.transform(X[tr][ite])
                scaler = StandardScaler().fit(raw_tr)
                Xa = scaler.transform(raw_tr)
                Xb = scaler.transform(raw_te)
                clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(Xa, y[tr][itr])
                scores.append(accuracy_score(y[tr][ite], clf.predict(Xb)))
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_k, best_score = k, mean_score
        selected_counts.append(best_k)
        sel = make_selector(best_k).fit(X[tr], y[tr])
        raw_tr = sel.transform(X[tr])
        raw_te = sel.transform(X[te])
        scaler = StandardScaler().fit(raw_tr)
        Xtr_s = scaler.transform(raw_tr)
        Xte_s = scaler.transform(raw_te)
        k_actual = Xtr_s.shape[1]
        clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(Xtr_s, y[tr])
        pred = clf.predict(Xte_s)
        metrics["LogisticRegression"]["acc"].append(accuracy_score(y[te], pred))
        metrics["LogisticRegression"]["f1"].append(f1_score(y[te], pred, average="macro"))

        scores = jsd_scores(Xtr_s, y[tr])
        order = np.argsort(scores)[::-1]
        imgs_tr = make_images(Xtr_s[:, order], np.arange(k_actual), 20)
        imgs_te = make_images(Xte_s[:, order], np.arange(k_actual), 20)
        prob = train_torch_bce(FullSizeCNN(20, 4), imgs_tr, y[tr], imgs_te, binary=False)
        pred = prob.argmax(axis=1)
        metrics["FullSizeCNN"]["acc"].append(accuracy_score(y[te], pred))
        metrics["FullSizeCNN"]["f1"].append(f1_score(y[te], pred, average="macro"))

        prob = train_torch_bce(DenseBaseline(k_actual, 4), Xtr_s, y[tr], Xte_s, binary=False)
        pred = prob.argmax(axis=1)
        metrics["Dense"]["acc"].append(accuracy_score(y[te], pred))
        metrics["Dense"]["f1"].append(f1_score(y[te], pred, average="macro"))
        print("outer done", len(selected_counts), "best_k", best_k, flush=True)

    rows = []
    for model, m in metrics.items():
        rows.append({"experiment": "nested_cv_PAM50_mRNA", "model": model, "metric": "accuracy", "mean": round(float(np.mean(m["acc"])), 4), "std": round(float(np.std(m["acc"])), 4)})
        rows.append({"experiment": "nested_cv_PAM50_mRNA", "model": model, "metric": "macro_f1", "mean": round(float(np.mean(m["f1"])), 4), "std": round(float(np.std(m["f1"])), 4)})
    return rows, selected_counts


def cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    n = risk.shape[0]
    loss = torch.zeros((), dtype=risk.dtype, device=risk.device)
    for i in range(n):
        if event[i] == 0:
            continue
        risk_i = risk[i]
        at_risk = (time >= time[i]).float()
        logsum = torch.logsumexp(risk + torch.log(at_risk + 1e-12), dim=0)
        loss = loss + logsum - risk_i
    return loss / max(1, int(event.sum().item()))


def train_fullsize_cnn_cox(Xtr, ytr_time, ytr_event, Xte):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeCNNCox(Xtr.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    time = torch.tensor(ytr_time, dtype=torch.float32)
    event = torch.tensor(ytr_event, dtype=torch.float32)
    n = Xt.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if idx.shape[0] < 2:
                continue
            if event[idx].sum().item() == 0:
                continue
            opt.zero_grad()
            risk = model(Xt[idx]).view(-1)
            loss = cox_loss(risk, time[idx], event[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xte, dtype=torch.float32)).numpy().ravel()


def fullsize_cnn_cox():
    df = pd.read_csv(SUR_FINAL, sep="\t")
    X = df.drop(columns=["case_id", "os_event", "os_time_days"]).to_numpy(dtype=float)
    y_event = df["os_event"].astype(int).values
    y_time = df["os_time_days"].astype(float).values
    imgs = np.load(SUR_IMG)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs, cis = [], []
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        risk = train_fullsize_cnn_cox(imgs[tr], y_time[tr], y_event[tr], imgs[te])
        aucs.append(roc_auc_score(y_event[te], risk))
        cis.append(concordance_index(y_time[te], -risk, y_event[te]))
    return [{"experiment": "FullSizeCNN_Cox", "model": "FullSizeCNN_Cox", "metric": "roc_auc", "mean": round(float(np.mean(aucs)), 4), "std": round(float(np.std(aucs)), 4)},
            {"experiment": "FullSizeCNN_Cox", "model": "FullSizeCNN_Cox", "metric": "c_index", "mean": round(float(np.mean(cis)), 4), "std": round(float(np.std(cis)), 4)}]


def strict_pam50_exclusion():
    expr = pd.read_csv(HISEQ, sep="\t", index_col=0).T
    labels = pd.read_csv(DATA / "brca_labels_modeling_ready.tsv", sep="\t")
    labels = labels[labels["pam50_4class"].isin(["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"])]
    labels = labels.dropna(subset=["mrna_sample_id"])
    labels["prefix"] = labels["mrna_sample_id"].str[:15]
    idx = [s for s in labels["prefix"] if s in expr.index]
    X_full = expr.loc[idx].to_numpy(dtype=float)
    genes = np.array(expr.columns)
    keep = [i for i, g in enumerate(genes) if g not in PAM50_GENES]
    X_excl = X_full[:, keep]
    y_raw = labels.set_index("prefix").loc[idx, "pam50_4class"].values
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, X in [("full_transcriptome", X_full), ("exclude_pam50_genes", X_excl)]:
        accs, f1s = [], []
        for tr, te in cv.split(np.zeros(len(y)), y):
            sel = SelectKBest(f_classif, k=400).fit(X[tr], y[tr])
            raw_tr = sel.transform(X[tr])
            raw_te = sel.transform(X[te])
            scaler = StandardScaler().fit(raw_tr)
            Xtr = scaler.transform(raw_tr)
            Xte = scaler.transform(raw_te)
            clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(Xtr, y[tr])
            pred = clf.predict(Xte)
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="macro"))
        rows.append({"experiment": "strict_PAM50_exclusion", "config": name, "metric": "accuracy", "mean": round(float(np.mean(accs)), 4), "std": round(float(np.std(accs)), 4)})
        rows.append({"experiment": "strict_PAM50_exclusion", "config": name, "metric": "macro_f1", "mean": round(float(np.mean(f1s)), 4), "std": round(float(np.std(f1s)), 4)})
    return rows


def write_tsv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def main():
    if not OUT_NESTED.exists():
        nested_rows, counts = nested_pam50_mrna()
        write_tsv(OUT_NESTED, nested_rows, ["experiment", "model", "metric", "mean", "std"])
    else:
        nested_rows = []
        counts = []
    cox_rows = fullsize_cnn_cox()
    write_tsv(OUT_COX, cox_rows, ["experiment", "model", "metric", "mean", "std"])
    excl_rows = strict_pam50_exclusion()
    write_tsv(OUT_EXCL, excl_rows, ["experiment", "config", "metric", "mean", "std"])
    print("selected k counts", pd.Series(counts).value_counts().to_dict())
    print("done")


if __name__ == "__main__":
    main()
