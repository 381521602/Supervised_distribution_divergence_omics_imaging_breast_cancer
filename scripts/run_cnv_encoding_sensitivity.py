# -*- coding: utf-8 -*-
"""CNV encoding sensitivity: rebuilt integer CNV vs GISTIC2 thresholded.

Runs the same leak-free fold-internal pipeline on the rebuilt integer CNV
matrix (data/cnv_gene_matrix.tsv) so the results can be compared with the
GISTIC2-based CNV results in leakfree_single_omics_results.tsv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_leakfree_single_omics import (  # noqa: E402
    RANDOM_STATE, K_FOLDS, PAM50_K, SURV_K, PAM50_SIZE, SURV_SIZE,
    build_adaptive_pipeline, jsd_scores, make_images, ml_models, train_torch,
    MLP, FullSizeCNN, Dense,
)

LABELS = ROOT / "data" / "brca_labels_modeling_ready.tsv"
CNV_REBUILT = WORK / "data" / "cnv_gene_matrix.tsv"
OUT = ROOT / "data" / "results" / "leakfree_cnv_rebuilt_integer_results.tsv"


def load_rebuilt_cnv():
    raw = pd.read_csv(CNV_REBUILT, sep="\t", index_col=0)  # samples x genes
    labels = pd.read_csv(LABELS, sep="\t")
    labels = labels.dropna(subset=["cnv_gene_sample_id"])
    labels["prefix"] = labels["cnv_gene_sample_id"].str[:15]
    labels = labels.drop_duplicates(subset=["prefix"], keep="first")
    idx = [s for s in labels["prefix"] if s in raw.index]
    X = raw.loc[idx].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, labels.set_index("prefix").loc[idx].reset_index(drop=True)


def eval_pam50():
    X, lab = load_rebuilt_cnv()
    m = lab["pam50_4class"].isin(["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"])
    X = X[m.to_numpy()]
    y = LabelEncoder().fit_transform(lab.loc[m, "pam50_4class"].values)
    k, size = PAM50_K["CNV"], PAM50_SIZE["CNV"]
    names = list(ml_models().keys()) + ["MLP", "FullSizeCNN", "Dense"]
    acc, f1 = {n: [] for n in names}, {n: [] for n in names}
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in cv.split(np.zeros(len(y)), y):
        pipe = build_adaptive_pipeline("CNV", "PAM50_4class", k=k).fit(X[tr], y[tr])
        Xtr, Xte = pipe.transform(X[tr]), pipe.transform(X[te])
        order = np.argsort(jsd_scores(Xtr, y[tr]))[::-1]
        imgs_tr = make_images(Xtr[:, order], np.arange(Xtr.shape[1]), size)
        imgs_te = make_images(Xte[:, order], np.arange(Xte.shape[1]), size)
        for name, clf in ml_models().items():
            pred = clf.fit(Xtr, y[tr]).predict(Xte)
            acc[name].append(accuracy_score(y[te], pred))
            f1[name].append(f1_score(y[te], pred, average="macro"))
        p = train_torch(MLP(Xtr.shape[1], 4), Xtr, y[tr], Xte, binary=False).argmax(1)
        acc["MLP"].append(accuracy_score(y[te], p)); f1["MLP"].append(f1_score(y[te], p, average="macro"))
        p = train_torch(FullSizeCNN(size, 4), imgs_tr, y[tr], imgs_te, binary=False).argmax(1)
        acc["FullSizeCNN"].append(accuracy_score(y[te], p)); f1["FullSizeCNN"].append(f1_score(y[te], p, average="macro"))
        p = train_torch(Dense(Xtr.shape[1], 4), Xtr, y[tr], Xte, binary=False).argmax(1)
        acc["Dense"].append(accuracy_score(y[te], p)); f1["Dense"].append(f1_score(y[te], p, average="macro"))
        print("PAM50 fold done", flush=True)
    rows = []
    for n in names:
        rows += [{"task": "PAM50", "model": n, "metric": "accuracy", "mean": round(float(np.mean(acc[n])), 4), "std": round(float(np.std(acc[n])), 4)},
                 {"task": "PAM50", "model": n, "metric": "macro_f1", "mean": round(float(np.mean(f1[n])), 4), "std": round(float(np.std(f1[n])), 4)}]
    return rows


def eval_survival():
    X, lab = load_rebuilt_cnv()
    m = lab["os_event"].isin([0, 1]) & lab["os_time_days"].notna()
    X = X[m.to_numpy()]
    y_event = lab.loc[m, "os_event"].astype(int).values
    y_time = lab.loc[m, "os_time_days"].astype(float).values
    k, size = SURV_K["CNV"], SURV_SIZE["CNV"]
    names = list(ml_models().keys()) + ["MLP", "FullSizeCNN", "Dense", "CoxPH"]
    auc, ci = {n: [] for n in names}, {n: [] for n in names}
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in cv.split(np.zeros(len(y_event)), y_event):
        pipe = build_adaptive_pipeline("CNV", "OS", k=k).fit(X[tr], y_event[tr])
        Xtr, Xte = pipe.transform(X[tr]), pipe.transform(X[te])
        order = np.argsort(jsd_scores(Xtr, y_event[tr]))[::-1]
        imgs_tr = make_images(Xtr[:, order], np.arange(Xtr.shape[1]), size)
        imgs_te = make_images(Xte[:, order], np.arange(Xte.shape[1]), size)
        for name, clf in ml_models().items():
            mdl = clf.fit(Xtr, y_event[tr])
            risk = mdl.predict_proba(Xte)[:, 1] if hasattr(mdl, "predict_proba") else mdl.decision_function(Xte)
            auc[name].append(roc_auc_score(y_event[te], risk)); ci[name].append(concordance_index(y_time[te], -risk, y_event[te]))
        risk = train_torch(MLP(Xtr.shape[1], 1), Xtr, y_event[tr], Xte, binary=True)
        auc["MLP"].append(roc_auc_score(y_event[te], risk)); ci["MLP"].append(concordance_index(y_time[te], -risk, y_event[te]))
        risk = train_torch(FullSizeCNN(size, 1), imgs_tr, y_event[tr], imgs_te, binary=True)
        auc["FullSizeCNN"].append(roc_auc_score(y_event[te], risk)); ci["FullSizeCNN"].append(concordance_index(y_time[te], -risk, y_event[te]))
        risk = train_torch(Dense(Xtr.shape[1], 1), Xtr, y_event[tr], Xte, binary=True)
        auc["Dense"].append(roc_auc_score(y_event[te], risk)); ci["Dense"].append(concordance_index(y_time[te], -risk, y_event[te]))
        print("Survival fold done", flush=True)
    rows = []
    for n in names:
        rows += [{"task": "Survival", "model": n, "metric": "roc_auc", "mean": round(float(np.mean(auc[n])), 4), "std": round(float(np.std(auc[n])), 4)},
                 {"task": "Survival", "model": n, "metric": "c_index", "mean": round(float(np.mean(ci[n])), 4), "std": round(float(np.std(ci[n])), 4)}]
    return rows


def main():
    rows = eval_pam50() + eval_survival()
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "model", "metric", "mean", "std"], delimiter="\t")
        w.writeheader(); w.writerows(rows)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
