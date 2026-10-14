# -*- coding: utf-8 -*-
"""Bootstrap 95% CI for key METABRIC external-validation metrics."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"E:\组学数据图像化")
sys.path.insert(0, str(ROOT / "scripts"))
from run_metabric_external_validation import (
    load_metabric_clinical,
    load_tcga_pam50,
    load_tcga_survival,
    prepare_metabric_labels,
)

DATA = ROOT / "data"
MATRIX = DATA / "external" / "metabric" / "metabric_mrna_zscores_selected.tsv"
RNG = np.random.default_rng(42)


def bootstrap_ci(metric_fn, y_true, y_score, n_boot=2000):
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    matrix_symbols = pd.read_csv(MATRIX, sep="\t", index_col=0)
    clinical = load_metabric_clinical()
    pam_valid, survival = prepare_metabric_labels(clinical, matrix_symbols)

    # PAM50
    y_pam, pam_genes, X_pam = load_tcga_pam50()
    pam_common = [g for g in pam_genes if g in matrix_symbols.index]
    idx = [pam_genes.index(g) for g in pam_common]
    Xtr = np.nan_to_num(X_pam[:, idx], nan=0.0)
    Xm = np.nan_to_num(matrix_symbols.loc[pam_common].T.to_numpy(dtype=float), nan=0.0)
    meta_s = Xm  # all 1980; subset below
    meta_pam_s = matrix_symbols.columns.intersection(pam_valid.index)
    pos = {c: i for i, c in enumerate(matrix_symbols.columns)}
    meta_idx = [pos[c] for c in meta_pam_s]
    Xte = meta_s[meta_idx]
    yte = pam_valid.loc[meta_pam_s].values
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), StandardScaler().fit(Xte).transform(Xte)
    for name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=3000, random_state=42)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
    ]:
        model.fit(Xtr_s, y_pam)
        pred = model.predict(Xte_s)
        acc = accuracy_score(yte, pred)
        # bootstrap on hard predictions
        lo, hi = bootstrap_ci(lambda a, b: accuracy_score(a, b), yte, pred)
        print(f"PAM50 {name}: n={len(yte)} acc={acc:.4f} 95%CI=[{lo:.4f},{hi:.4f}]")

    # Survival
    tcga_sur, sur_genes, X_sur, sur_time, sur_event = load_tcga_survival()
    sur_common = [g for g in sur_genes if g in matrix_symbols.index]
    Xtr_sur = np.nan_to_num(tcga_sur[sur_common].to_numpy(dtype=float), nan=0.0)
    y_event = tcga_sur["os_event"].astype(int).values
    Xm_sur = np.nan_to_num(matrix_symbols.loc[sur_common].T.to_numpy(dtype=float), nan=0.0)
    meta_sur_s = matrix_symbols.columns.intersection(survival.index)
    pos2 = {c: i for i, c in enumerate(matrix_symbols.columns)}
    meta_idx2 = [pos2[c] for c in meta_sur_s]
    Xte_sur = Xm_sur[meta_idx2]
    meta_sur = survival.loc[meta_sur_s]
    meta_time = meta_sur["os_months"].astype(float).values * 30.44
    meta_event = meta_sur["os_event"].astype(int).values
    sc2 = StandardScaler().fit(Xtr_sur)
    Xtr_sur_s, Xte_sur_s = sc2.transform(Xtr_sur), StandardScaler().fit(Xte_sur).transform(Xte_sur)
    for name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=3000, random_state=42)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
    ]:
        model.fit(Xtr_sur_s, y_event)
        prob = model.predict_proba(Xte_sur_s)[:, 1]
        ci_ = concordance_index(meta_time, -prob, meta_event)
        auc = roc_auc_score(meta_event, prob)
        lo_c, hi_c = bootstrap_ci(lambda t, s: concordance_index(t, -s, meta_event), meta_time, prob)
        lo_a, hi_a = bootstrap_ci(lambda a, b: roc_auc_score(a, b), meta_event, prob)
        print(f"Survival {name}: n={len(meta_event)} events={int(meta_event.sum())} "
              f"c_index={ci_:.4f} 95%CI=[{lo_c:.4f},{hi_c:.4f}] "
              f"roc_auc={auc:.4f} 95%CI=[{lo_a:.4f},{hi_a:.4f}]")


if __name__ == "__main__":
    main()
