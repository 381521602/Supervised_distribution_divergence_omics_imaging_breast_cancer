#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test an optimized JSD-based feature-selection pipeline.

Classification (PAM50):
  L1 coarse filter -> one-vs-rest JSD fine filter -> LogisticRegression/RandomForest

Survival:
  L1 coarse filter -> binary JSD fine filter -> LogisticRegression / CoxPH
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_entropy_feature_selection import jsd_between_histograms  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
OUT = DATA / "optimal_jsd_results.tsv"
MATRICES = {
    "mRNA": DATA / "external" / "xena" / "HiSeqV2",
    "CNV": DATA / "external" / "xena" / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
    "miRNA": DATA / "external" / "xena" / "miRNA_HiSeq_gene",
}

RANDOM_STATE = 42
K_FOLDS = 5
N_BINS = 10
EPS = 1e-9


class OvRJSDSelector(BaseEstimator, TransformerMixin):
    """One-vs-rest JSD: score a feature by max separation against each class."""

    def __init__(self, k: int = 200, n_bins: int = N_BINS):
        self.k = k
        self.n_bins = n_bins

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)
        scores = np.zeros(X.shape[1], dtype=float)

        for j in range(X.shape[1]):
            feature = X[:, j]
            finite = feature[np.isfinite(feature)]
            if finite.size < 10:
                continue
            qs = np.unique(np.quantile(finite, np.linspace(0, 1, self.n_bins + 1)))
            if qs.size < 2:
                continue
            bins = np.digitize(feature, qs[1:-1])
            best = 0.0
            for cls in classes:
                pos = np.bincount(bins[y == cls], minlength=self.n_bins)
                neg = np.bincount(bins[y != cls], minlength=self.n_bins)
                best = max(best, jsd_between_histograms(pos, neg))
            scores[j] = best

        self.selected_features_ = np.argsort(scores)[::-1][: self.k]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_features_]


def l1_selector(k: int):
    return SelectFromModel(
        LinearSVC(penalty="l1", dual=False, C=0.1, max_iter=5000, random_state=RANDOM_STATE),
        max_features=k,
        threshold=-np.inf,
    )


def fclassif_selector(k: int):
    return SelectKBest(score_func=f_classif, k=k)


def build_classifier(selector_kind: str):
    variance = VarianceThreshold(threshold=0.0)
    coarse = l1_selector(1000) if selector_kind == "L1" else fclassif_selector(1000)
    return Pipeline(
        [
            ("variance", variance),
            ("coarse", coarse),
            ("ovr_jsd", OvRJSDSelector(k=200)),
            ("scale", StandardScaler()),
        ]
    )


def cv_classification(omics: str, method: str, clf_name: str, clf, X, y) -> list[dict]:
    prep = build_classifier("L1" if method.startswith("L1") else "FClassif")
    model = Pipeline([("prep", prep), ("clf", clf)])
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1_macro", "balanced_accuracy"],
        n_jobs=1,
        error_score="raise",
    )
    return [
        {
            "omics": omics,
            "task": "PAM50_4class",
            "feature_method": method,
            "model": clf_name,
            "metric": "accuracy",
            "value": round(float(np.mean(scores["test_accuracy"])), 4),
        },
        {
            "omics": omics,
            "task": "PAM50_4class",
            "feature_method": method,
            "model": clf_name,
            "metric": "macro_f1",
            "value": round(float(np.mean(scores["test_f1_macro"])), 4),
        },
        {
            "omics": omics,
            "task": "PAM50_4class",
            "feature_method": method,
            "model": clf_name,
            "metric": "balanced_accuracy",
            "value": round(float(np.mean(scores["test_balanced_accuracy"])), 4),
        },
    ]


def cv_binary_auc(omics: str, X, y) -> list[dict]:
    prep = Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("l1", l1_selector(1000)),
            ("jsd", OvRJSDSelector(k=200)),
            ("scale", StandardScaler()),
        ]
    )
    model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc"], n_jobs=1, error_score="raise")
    return [
        {
            "omics": omics,
            "task": "OS_binary",
            "feature_method": "L1_OvRJSD",
            "model": "LogisticRegression",
            "metric": "roc_auc",
            "value": round(float(np.mean(scores["test_roc_auc"])), 4),
        }
    ]


def cv_cox_cindex(omics: str, X, y_time, y_event) -> list[dict]:
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    c_indices: list[float] = []
    for train_idx, test_idx in cv.split(X, y_event):
        prep = Pipeline(
            [
                ("variance", VarianceThreshold(threshold=0.0)),
                ("l1", l1_selector(1000)),
                ("jsd", OvRJSDSelector(k=50)),
                ("scale", StandardScaler()),
            ]
        )
        prep.fit(X[train_idx], y_event[train_idx])
        X_train = prep.transform(X[train_idx])
        X_test = prep.transform(X[test_idx])
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
            "feature_method": "L1_OvRJSD",
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
        for method, clf_name, clf in [
            ("L1_OvRJSD", "LogisticRegression", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ("L1_OvRJSD", "RandomForest", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
            ("FClassif_OvRJSD", "LogisticRegression", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]:
            try:
                rows.extend(cv_classification(omics, method, clf_name, clf, X, y))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {omics}/PAM50/{method}/{clf_name}: {exc}")

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
