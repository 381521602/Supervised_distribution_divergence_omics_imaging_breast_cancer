#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable adaptive JSD feature-selection module.

The module chooses a feature-selection chain per omics and task:
  - PAM50 mRNA : FClassif -> JSD
  - PAM50 CNV  : L1 -> One-vs-Rest JSD
  - PAM50 miRNA: JSD -> L1
  - OS all     : L1 -> JSD

It exposes select_features(), which returns the selected feature names and the
reduced sample x feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
N_BINS = 10
EPS = 1e-9


def jsd_between_histograms(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = (p + EPS) / (p.sum() + EPS * p.size)
    q = (q + EPS) / (q.sum() + EPS * q.size)
    denom = p + q
    return float(
        0.5 * np.sum(p * np.log2(2.0 * p / denom))
        + 0.5 * np.sum(q * np.log2(2.0 * q / denom))
    )


class JSDSelector(BaseEstimator, TransformerMixin):
    """Average pairwise JSD selector."""

    def __init__(self, prefilter_k: int = 2000, k: int = 200, n_bins: int = N_BINS):
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
            hists = [np.bincount(bins[y == cls], minlength=self.n_bins) for cls in classes]
            if len(hists) == 2:
                scores[j] = jsd_between_histograms(hists[0], hists[1])
            else:
                pair_scores = [
                    jsd_between_histograms(hists[a], hists[b])
                    for a in range(len(hists))
                    for b in range(a + 1, len(hists))
                ]
                scores[j] = float(np.mean(pair_scores)) if pair_scores else 0.0

        top_local = np.argsort(scores)[::-1][: self.k]
        self.selected_features_ = np.sort(pre_idx[top_local])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_features_]


class OvRJSDSelector(BaseEstimator, TransformerMixin):
    """One-vs-rest JSD selector."""

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


def fclassif_selector(k: int):
    return SelectKBest(score_func=f_classif, k=k)


def l1_selector(k: int):
    return SelectFromModel(
        LinearSVC(penalty="l1", dual=False, C=0.1, max_iter=5000, random_state=RANDOM_STATE),
        max_features=k,
        threshold=-np.inf,
    )


def build_adaptive_pipeline(omics: str, task: str, k: int = 200) -> Pipeline:
    """Build the adaptive preprocessing pipeline for a task and omics."""
    variance = VarianceThreshold(threshold=0.0)
    if task == "PAM50_4class":
        if omics == "mRNA":
            coarse = fclassif_selector(max(k * 5, 1000))
            fine = JSDSelector(prefilter_k=max(k * 5, 1000), k=k)
        elif omics == "CNV":
            coarse = l1_selector(max(k * 5, 1000))
            fine = OvRJSDSelector(k=k)
        elif omics == "miRNA":
            coarse = JSDSelector(prefilter_k=max(k * 5, 1000), k=max(k * 5, 1000))
            fine = l1_selector(k)
        else:
            coarse = fclassif_selector(max(k * 5, 1000))
            fine = JSDSelector(prefilter_k=max(k * 5, 1000), k=k)
    elif task == "OS":
        coarse = l1_selector(max(k * 5, 1000))
        fine = JSDSelector(prefilter_k=max(k * 5, 1000), k=k)
    else:
        raise ValueError(task)

    return Pipeline(
        [
            ("variance", variance),
            ("coarse", coarse),
            ("fine", fine),
            ("scale", StandardScaler()),
        ]
    )


def select_features(
    matrix: pd.DataFrame,
    labels: pd.DataFrame,
    omics: str,
    task: str,
    k: int = 200,
) -> tuple[list[str], pd.DataFrame]:
    """Fit the adaptive selector and return feature names + reduced matrix."""
    if task == "PAM50_4class":
        case_ids = labels.loc[
            labels["pam50_4class"].notna() & (labels["pam50_4class"] != ""),
            "case_id",
        ].tolist()
        y_col = "pam50_4class"
    elif task == "OS":
        case_ids = labels.loc[
            labels["os_event"].isin([0, 1]) & labels["os_time_days"].notna(),
            "case_id",
        ].tolist()
        y_col = "os_event"
    else:
        raise ValueError(task)

    X, cases = _align_matrix(matrix, case_ids)
    y = labels.set_index("case_id").loc[cases, y_col].values
    pipeline = build_adaptive_pipeline(omics, task, k=k)
    pipeline.fit(X, y)

    # Recover final selected feature indices through the fitted pipeline.
    selectors = [
        step
        for name, step in pipeline.steps
        if name in {"coarse", "fine"}
    ]
    selected_indices = np.arange(X.shape[1])
    for selector in selectors:
        indices = getattr(selector, "selected_features_", None)
        if indices is None:
            support = getattr(selector, "get_support", lambda: None)()
            if support is not None:
                indices = np.where(support)[0]
        if indices is not None:
            selected_indices = selected_indices[indices]

    feature_names = [str(matrix.index[i]) for i in selected_indices]
    reduced = pd.DataFrame(
        X[:, selected_indices],
        index=pd.Index(cases, name="case_id"),
        columns=pd.Index(feature_names, name="feature"),
    )
    return feature_names, reduced


def _align_matrix(matrix: pd.DataFrame, case_ids: list[str]):
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
