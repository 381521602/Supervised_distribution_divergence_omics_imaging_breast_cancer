#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSD (new symmetric relative entropy) feature selection + classical ML.

The project proposal uses JSD:
    JSD(p||q) = sum_i p_i * log2(2*p_i/(p_i+q_i))
               + sum_i q_i * log2(2*q_i/(p_i+q_i))

Here each feature is discretized into quantile bins using the training fold,
class-conditional histograms are estimated, and JSD is averaged over all class
pairs (PAM50) or computed between the two classes (survival event).

Outputs: data/entropy_feature_selection_results.tsv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
OUT = DATA / "entropy_feature_selection_results.tsv"

MATRICES = {
    "mRNA": DATA / "external" / "xena" / "HiSeqV2",
    "CNV": DATA / "external" / "xena" / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
    "miRNA": DATA / "external" / "xena" / "miRNA_HiSeq_gene",
}

RANDOM_STATE = 42
K_FOLDS = 5
N_BINS = 10
PREFILTER_K = 2000
N_FEATURES = 200
EPS = 1e-9


def jsd_between_histograms(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = (p + EPS) / (p.sum() + EPS * p.size)
    q = (q + EPS) / (q.sum() + EPS * q.size)
    denom = p + q
    return float(
        np.sum(p * np.log2(2.0 * p / denom))
        + np.sum(q * np.log2(2.0 * q / denom))
    )


class JSDSelector(BaseEstimator, TransformerMixin):
    """Select top-k features by variance prefilter then JSD."""

    def __init__(self, prefilter_k: int = PREFILTER_K, k: int = N_FEATURES, n_bins: int = N_BINS):
        self.prefilter_k = prefilter_k
        self.k = k
        self.n_bins = n_bins

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)

        variances = np.var(X, axis=0)
        pre_idx = np.argsort(variances)[::-1][: self.prefilter_k]
        X_pre = X[:, pre_idx]

        scores = np.zeros(X_pre.shape[1], dtype=float)
        for j in range(X_pre.shape[1]):
            feature = X_pre[:, j]
            finite = feature[np.isfinite(feature)]
            if finite.size < 10:
                continue
            qs = np.unique(np.quantile(finite, np.linspace(0, 1, self.n_bins + 1)))
            if qs.size < 2:
                continue
            bins = np.digitize(feature, qs[1:-1])
            hists = []
            for cls in classes:
                counts = np.bincount(bins[y == cls], minlength=self.n_bins)
                hists.append(counts)

            if len(hists) == 2:
                scores[j] = jsd_between_histograms(hists[0], hists[1])
            else:
                pair_scores = []
                for a in range(len(hists)):
                    for b in range(a + 1, len(hists)):
                        pair_scores.append(jsd_between_histograms(hists[a], hists[b]))
                scores[j] = float(np.mean(pair_scores)) if pair_scores else 0.0

        top_local = np.argsort(scores)[::-1][: self.k]
        self.selected_features_ = np.sort(pre_idx[top_local])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_features_]


def cv_classification(omics: str, X, y) -> list[dict]:
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict] = []
    for clf_name, clf in [
        ("LogisticRegression", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
    ]:
        model = Pipeline(
            [
                ("entropy", JSDSelector()),
                ("scale", StandardScaler()),
                ("clf", clf),
            ]
        )
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "f1_macro", "balanced_accuracy"],
            n_jobs=1,
            error_score="raise",
        )
        rows.append(
            {
                "omics": omics,
                "task": "PAM50_4class",
                "feature_method": "JSD",
                "model": clf_name,
                "metric": "accuracy",
                "value": round(float(np.mean(scores["test_accuracy"])), 4),
            }
        )
        rows.append(
            {
                "omics": omics,
                "task": "PAM50_4class",
                "feature_method": "JSD",
                "model": clf_name,
                "metric": "macro_f1",
                "value": round(float(np.mean(scores["test_f1_macro"])), 4),
            }
        )
        rows.append(
            {
                "omics": omics,
                "task": "PAM50_4class",
                "feature_method": "JSD",
                "model": clf_name,
                "metric": "balanced_accuracy",
                "value": round(float(np.mean(scores["test_balanced_accuracy"])), 4),
            }
        )
    return rows


def cv_binary_auc(omics: str, X, y) -> list[dict]:
    model = Pipeline(
        [
            ("entropy", JSDSelector()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc"], n_jobs=1, error_score="raise")
    return [
        {
            "omics": omics,
            "task": "OS_binary",
            "feature_method": "JSD",
            "model": "LogisticRegression",
            "metric": "roc_auc",
            "value": round(float(np.mean(scores["test_roc_auc"])), 4),
        }
    ]


def cv_cox_cindex(omics: str, X, y_time, y_event) -> list[dict]:
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    c_indices: list[float] = []
    for train_idx, test_idx in cv.split(X, y_event):
        selector = JSDSelector(prefilter_k=PREFILTER_K, k=50)
        selector.fit(X[train_idx], y_event[train_idx])
        X_train = selector.transform(X[train_idx])
        X_test = selector.transform(X[test_idx])
        train_df = pd.DataFrame(X_train)
        train_df["time"] = y_time[train_idx]
        train_df["event"] = y_event[train_idx].astype(int)
        test_df = pd.DataFrame(X_test)
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_df, duration_col="time", event_col="event")
        hazard = cph.predict_partial_hazard(test_df)
        c_indices.append(float(concordance_index(y_time[test_idx], -hazard, y_event[test_idx])))
    return [
        {
            "omics": omics,
            "task": "OS_Cox",
            "feature_method": "JSD",
            "model": "CoxPH",
            "metric": "c_index",
            "value": round(float(np.mean(c_indices)), 4),
        }
    ]


def align_matrix(matrix: pd.DataFrame, case_ids: list[str]):
    case_to_cols: dict[str, list[str]] = {}
    for col in matrix.columns:
        case = col[:12]
        case_to_cols.setdefault(case, []).append(col)
    case_to_best = {
        case: (([c for c in cols if c.endswith("-01")] or cols)[0])
        for case, cols in case_to_cols.items()
    }
    aligned_cases = [c for c in case_ids if c in case_to_best]
    cols = [case_to_best[c] for c in aligned_cases]
    X = matrix[cols].T.to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(aligned_cases)


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    rows: list[dict] = []

    for omics, path in MATRICES.items():
        if not path.exists():
            continue
        matrix = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
        matrix = matrix.loc[:, ~matrix.columns.duplicated()]

        pam_cases = labels.loc[labels["pam50_4class"].notna() & (labels["pam50_4class"] != ""), "case_id"].tolist()
        X, cases = align_matrix(matrix, pam_cases)
        y = labels.set_index("case_id").loc[cases, "pam50_4class"].values
        rows.extend(cv_classification(omics, X, y))

        os_cases = labels.loc[
            labels["os_event"].isin([0, 1]) & labels["os_time_days"].notna(),
            "case_id",
        ].tolist()
        X, cases = align_matrix(matrix, os_cases)
        y_event = labels.set_index("case_id").loc[cases, "os_event"].astype(int).values
        y_time = labels.set_index("case_id").loc[cases, "os_time_days"].astype(float).values
        rows.extend(cv_binary_auc(omics, X, y_event))
        rows.extend(cv_cox_cindex(omics, X, y_time, y_event))

    if rows:
        columns = ["omics", "task", "feature_method", "model", "metric", "value"]
        with OUT.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} result rows -> {OUT}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
