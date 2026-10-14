# -*- coding: utf-8 -*-
"""Leak-free stacking paired tests against best single omics and fusion.

All models are evaluated on the identical 5 outer folds in a single pass so
that fold-level scores are directly paired. Paired t-test and Wilcoxon
signed-rank test are computed with scipy.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
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

OUT_SCORES = ROOT / "data" / "leakfree_stacking_fold_scores.tsv"
OUT_TESTS = ROOT / "data" / "leakfree_stacking_paired_tests.tsv"
OUTER = 5
INNER = 5
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
    compare = (["mRNA_RF", "miRNA_RF"] if binary else ["mRNA_RF"])
    compare += ["CNN_LateAvg", "Concat_MLP"]
    all_names = compare + ["Stacking_9"]
    cv = StratifiedKFold(n_splits=OUTER, shuffle=True, random_state=RANDOM_STATE)

    acc = {n: [] for n in all_names}
    f1c = {n: [] for n in all_names}

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

        # single-omics RF baselines
        for o, key in (("mRNA", "mRNA_RF"), ("miRNA", "miRNA_RF")):
            if not binary and o == "miRNA":
                continue
            Xtr_o, Xte_o = feat[o]
            rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr_o, y[tr])
            prob = rf.predict_proba(Xte_o)[:, 1] if binary else rf.predict_proba(Xte_o)
            if binary:
                acc[key].append(roc_auc_score(y[te], prob))
                f1c[key].append(concordance_index(yt[te], -prob, y[te]))
            else:
                acc[key].append(accuracy_score(y[te], prob.argmax(1)))
                f1c[key].append(f1_score(y[te], prob.argmax(1), average="macro"))

        # fusion models on concatenated features / late-average images
        Xtr_c = np.hstack([feat[o][0] for o in OMICS])
        Xte_c = np.hstack([feat[o][1] for o in OMICS])
        p_mlp = train_mlp(Xtr_c, y[tr], Xte_c, binary)
        probs = [train_single_prob(img[o][0], y[tr], img[o][1], SZ[o], binary) for o in OMICS]
        late = sum(probs) / len(probs)
        for key, prob in (("Concat_MLP", p_mlp), ("CNN_LateAvg", late)):
            if binary:
                acc[key].append(roc_auc_score(y[te], prob))
                f1c[key].append(concordance_index(yt[te], -prob, y[te]))
            else:
                acc[key].append(accuracy_score(y[te], prob.argmax(1)))
                f1c[key].append(f1_score(y[te], prob.argmax(1), average="macro"))

        # stacking (inner OOF + meta-learner)
        oof = {n: (np.zeros(len(tr)) if binary else np.zeros((len(tr), n_classes))) for n in base_names}
        inner = StratifiedKFold(n_splits=INNER, shuffle=True, random_state=RANDOM_STATE)
        for itr_i, ite_i in inner.split(np.zeros(len(tr)), y[tr]):
            for o in OMICS:
                Xtr_o, _ = feat[o]
                itr_o, _ = img[o]
                clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(Xtr_o[itr_i], y[tr][itr_i])
                oof[f"{o}_ML"][ite_i] = clf.predict_proba(Xtr_o[ite_i])[:, 1] if binary else clf.predict_proba(Xtr_o[ite_i])
                oof[f"{o}_MLP"][ite_i] = train_mlp(Xtr_o[itr_i], y[tr][itr_i], Xtr_o[ite_i], binary)
                oof[f"{o}_CNN"][ite_i] = train_single_prob(itr_o[itr_i], y[tr][itr_i], itr_o[ite_i], SZ[o], binary)
        test_pred = {}
        for o in OMICS:
            Xtr_o, Xte_o = feat[o]
            itr_o, ite_o = img[o]
            clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(Xtr_o, y[tr])
            test_pred[f"{o}_ML"] = clf.predict_proba(Xte_o)[:, 1] if binary else clf.predict_proba(Xte_o)
            test_pred[f"{o}_MLP"] = train_mlp(Xtr_o, y[tr], Xte_o, binary)
            test_pred[f"{o}_CNN"] = train_single_prob(itr_o, y[tr], ite_o, SZ[o], binary)
        Xtr_meta = (np.column_stack([oof[n] for n in base_names]) if binary
                    else np.hstack([oof[n] for n in base_names]))
        Xte_meta = (np.column_stack([test_pred[n] for n in base_names]) if binary
                    else np.hstack([test_pred[n] for n in base_names]))
        meta = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE).fit(Xtr_meta, y[tr])
        sp = meta.predict_proba(Xte_meta)[:, 1] if binary else meta.predict_proba(Xte_meta)
        if binary:
            acc["Stacking_9"].append(roc_auc_score(y[te], sp))
            f1c["Stacking_9"].append(concordance_index(yt[te], -sp, y[te]))
        else:
            acc["Stacking_9"].append(accuracy_score(y[te], sp.argmax(1)))
            f1c["Stacking_9"].append(f1_score(y[te], sp.argmax(1), average="macro"))
        print(f"[{tname} fold] done", flush=True)

    m1 = "roc_auc" if binary else "accuracy"
    m2 = "c_index" if binary else "macro_f1"
    score_rows = []
    for n in all_names:
        for i in range(OUTER):
            score_rows.append([tname, n, m1, i, round(float(acc[n][i]), 5)])
            score_rows.append([tname, n, m2, i, round(float(f1c[n][i]), 5)])
    test_rows = []
    for base in compare:
        for metric_name, arr in ((m1, acc),):
            d = np.array(arr["Stacking_9"]) - np.array(arr[base])
            t, p = stats.ttest_rel(arr["Stacking_9"], arr[base])
            try:
                _, wp = stats.wilcoxon(arr["Stacking_9"], arr[base])
            except ValueError:
                wp = float("nan")
            test_rows.append([tname, "Stacking_9", base, metric_name,
                              round(float(np.mean(arr["Stacking_9"])), 4),
                              round(float(np.mean(arr[base])), 4),
                              round(float(np.mean(d)), 4), round(float(p), 4), round(float(wp), 4), len(cases)])
    return score_rows, test_rows


def main():
    score_rows, test_rows = [], []
    for task in ("PAM50_4class", "OS"):
        s, t = run_task(task)
        score_rows += s
        test_rows += t
    with OUT_SCORES.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["task", "model", "metric", "fold", "score"])
        w.writerows(score_rows)
    with OUT_TESTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["task", "model_a", "model_b", "metric", "a_mean", "b_mean", "diff", "ttest_p", "wilcoxon_p", "n"])
        w.writerows(test_rows)
    print("wrote", OUT_SCORES, "and", OUT_TESTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
