# -*- coding: utf-8 -*-
"""Leak-free 9-model stacking on the three-omics intersection.

Feature selection, JSD ordering and image generation are performed strictly
inside each outer training fold. Inner 5-fold stratified CV generates the
out-of-fold base-model predictions used as meta-features; a LogisticRegression
meta-learner is fit on those OOF features. Metrics are averaged over the 5
outer folds.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_full_s5_s7 import (
    align,
    fold_sel,
    jsd_scores,
    load_omics,
    make_images,
    train_mlp,
    OMICS,
    PAM50_K,
    PAM50_SIZE,
    RANDOM_STATE,
    SURV_K,
    SURV_SIZE,
)
from run_fullsize_multimodal_integration import train_single_prob

OUT = ROOT / "data" / "leakfree_stacking_results.tsv"
OUTER_FOLDS = 5
INNER_FOLDS = 5
KINDS = ["ML", "MLP", "CNN"]


def run_task(task: str):
    data, labels = load_omics(task)
    X, y, yt, cases = align(data, labels, task, OMICS)
    binary = task == "OS"
    n_classes = 1 if binary else 4
    K = PAM50_K if task == "PAM50_4class" else SURV_K
    SZ = PAM50_SIZE if task == "PAM50_4class" else SURV_SIZE
    tname = "PAM50" if task == "PAM50_4class" else "Survival"

    base_names = [f"{o}_{kind}" for o in OMICS for kind in KINDS]
    cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    m1 = {n: [] for n in base_names}
    m2 = {n: [] for n in base_names}
    s1, s2 = [], []

    for tr, te in cv.split(np.zeros(len(y)), y):
        feat = {}
        img = {}
        for o in OMICS:
            pipe = fold_sel(X, y, task, o, tr)
            Xtr_o = pipe.transform(X[o][tr])
            Xte_o = pipe.transform(X[o][te])
            order = np.argsort(jsd_scores(Xtr_o, y[tr]))[::-1]
            itr_o = make_images(Xtr_o[:, order], np.arange(Xtr_o.shape[1]), SZ[o])
            ite_o = make_images(Xte_o[:, order], np.arange(Xte_o.shape[1]), SZ[o])
            feat[o] = (Xtr_o, Xte_o)
            img[o] = (itr_o, ite_o)

        oof = {
            n: (np.zeros(len(tr)) if binary else np.zeros((len(tr), n_classes)))
            for n in base_names
        }
        inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for itr_i, ite_i in inner.split(np.zeros(len(tr)), y[tr]):
            for o in OMICS:
                Xtr_o, _ = feat[o]
                itr_o, _ = img[o]
                clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(
                    Xtr_o[itr_i], y[tr][itr_i]
                )
                if binary:
                    oof[f"{o}_ML"][ite_i] = clf.predict_proba(Xtr_o[ite_i])[:, 1]
                else:
                    oof[f"{o}_ML"][ite_i] = clf.predict_proba(Xtr_o[ite_i])
                oof[f"{o}_MLP"][ite_i] = train_mlp(
                    Xtr_o[itr_i], y[tr][itr_i], Xtr_o[ite_i], binary
                )
                oof[f"{o}_CNN"][ite_i] = train_single_prob(
                    itr_o[itr_i], y[tr][itr_i], itr_o[ite_i], SZ[o], binary
                )

        test_pred = {}
        for o in OMICS:
            Xtr_o, Xte_o = feat[o]
            itr_o, ite_o = img[o]
            clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(Xtr_o, y[tr])
            if binary:
                test_pred[f"{o}_ML"] = clf.predict_proba(Xte_o)[:, 1]
            else:
                test_pred[f"{o}_ML"] = clf.predict_proba(Xte_o)
            test_pred[f"{o}_MLP"] = train_mlp(Xtr_o, y[tr], Xte_o, binary)
            test_pred[f"{o}_CNN"] = train_single_prob(itr_o, y[tr], ite_o, SZ[o], binary)

        if binary:
            Xtr_meta = np.column_stack([oof[n] for n in base_names])
            Xte_meta = np.column_stack([test_pred[n] for n in base_names])
        else:
            Xtr_meta = np.hstack([oof[n] for n in base_names])
            Xte_meta = np.hstack([test_pred[n] for n in base_names])
        meta = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE).fit(Xtr_meta, y[tr])
        if binary:
            stack_prob = meta.predict_proba(Xte_meta)[:, 1]
        else:
            stack_prob = meta.predict_proba(Xte_meta)

        for n in base_names:
            prob = test_pred[n]
            if binary:
                m1[n].append(roc_auc_score(y[te], prob))
                m2[n].append(concordance_index(yt[te], -prob, y[te]))
            else:
                pred = prob.argmax(axis=1)
                m1[n].append(accuracy_score(y[te], pred))
                m2[n].append(f1_score(y[te], pred, average="macro"))
        if binary:
            s1.append(roc_auc_score(y[te], stack_prob))
            s2.append(concordance_index(yt[te], -stack_prob, y[te]))
        else:
            sp = stack_prob.argmax(axis=1)
            s1.append(accuracy_score(y[te], sp))
            s2.append(f1_score(y[te], sp, average="macro"))
        print(f"[{tname} outer fold] done", flush=True)

    rows = []
    for n in base_names:
        rows.append([n, tname, "accuracy" if not binary else "roc_auc",
                     round(float(np.mean(m1[n])), 4), round(float(np.std(m1[n])), 4), len(cases)])
        rows.append([n, tname, "macro_f1" if not binary else "c_index",
                     round(float(np.mean(m2[n])), 4), round(float(np.std(m2[n])), 4), len(cases)])
    rows.append(["Stacking_9", tname, "accuracy" if not binary else "roc_auc",
                 round(float(np.mean(s1)), 4), round(float(np.std(s1)), 4), len(cases)])
    rows.append(["Stacking_9", tname, "macro_f1" if not binary else "c_index",
                 round(float(np.mean(s2)), 4), round(float(np.std(s2)), 4), len(cases)])
    return rows


def main():
    rows = run_task("PAM50_4class") + run_task("OS")
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["model", "task", "metric", "mean", "std", "n"])
        w.writerows(rows)
    print("wrote", OUT, "rows", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
