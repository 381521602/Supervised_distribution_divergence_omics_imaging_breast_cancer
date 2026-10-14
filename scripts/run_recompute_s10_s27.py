# -*- coding: utf-8 -*-
"""Recompute S10 survival baseline (vs best single omics) and S27 effect sizes
from the leak-free same-fold predictions."""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
FOLDS = ROOT / "data" / "leakfree_stacking_fold_scores.tsv"
OUT_S10 = ROOT / "data" / "leakfree_s10_survival_paired.tsv"
OUT_S27 = ROOT / "data" / "leakfree_s27_effect_sizes.tsv"

df = pd.read_csv(FOLDS, sep="\t")


def fold_vec(task, model, metric):
    sub = df[(df["task"] == task) & (df["model"] == model) & (df["metric"] == metric)]
    return sub.sort_values("fold")["score"].to_numpy(dtype=float)


def paired_stats(a, b):
    d = a - b
    t, p = stats.ttest_rel(a, b)
    try:
        _, wp = stats.wilcoxon(a, b)
    except ValueError:
        wp = float("nan")
    n = len(d)
    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci_lo, ci_hi = md - tcrit * se, md + tcrit * se
    cohens_d = md / sd if sd > 0 else 0.0
    return md, p, wp, ci_lo, ci_hi, cohens_d, n


# S10: fusion vs best single-omics on the same folds
s10_rows = []
comparisons = [
    ("Survival", "CNN_LateAvg", "miRNA_RF", "roc_auc"),
    ("Survival", "Concat_MLP", "miRNA_RF", "roc_auc"),
    ("PAM50", "CNN_LateAvg", "mRNA_RF", "accuracy"),
    ("PAM50", "Concat_MLP", "mRNA_RF", "accuracy"),
]
for task, fusion, single, metric in comparisons:
    a = fold_vec(task, fusion, metric)
    b = fold_vec(task, single, metric)
    md, p, wp, ci_lo, ci_hi, d, n = paired_stats(a, b)
    s10_rows.append([task, fusion, single, metric, round(float(np.mean(a)), 4),
                     round(float(np.mean(b)), 4), round(md, 4), round(p, 4), round(wp, 4), n])

with OUT_S10.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["task", "fusion", "single", "metric", "fusion_mean", "single_mean", "mean_diff", "ttest_p", "wilcoxon_p", "n"])
    w.writerows(s10_rows)

# S27: effect sizes for the same comparisons
s27_rows = []
for task, fusion, single, metric in comparisons:
    a = fold_vec(task, fusion, metric)
    b = fold_vec(task, single, metric)
    md, p, wp, ci_lo, ci_hi, d, n = paired_stats(a, b)
    interp = "not significant" if p >= 0.05 else "significant"
    s27_rows.append([task, metric, f"{fusion}_vs_{single}", round(md, 4),
                     round(ci_lo, 4), round(ci_hi, 4), round(d, 4), interp])

with OUT_S27.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["task", "metric", "comparison", "mean_difference", "ci_lower", "ci_upper", "cohens_d", "interpretation"])
    w.writerows(s27_rows)

print("S10 rows:")
for r in s10_rows:
    print("  ", r)
print("S27 rows:")
for r in s27_rows:
    print("  ", r)
