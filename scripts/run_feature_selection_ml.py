#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare classical feature-selection/extraction methods before training.

Feature methods:
  - ANOVA F-value (SelectKBest)
  - Mutual information (SelectKBest)
  - L1 / LASSO logistic (SelectFromModel)
  - Random-forest importance (SelectFromModel)
  - Recursive feature elimination (RFE)
  - PCA (feature extraction)

After each method, LogisticRegression and RandomForest are trained with
5-fold stratified CV for PAM50 4-class classification. For overall survival,
the same preprocessing is used for a binary event classifier (AUC) and a
Cox model (C-index).

Outputs: data/feature_selection_ml_results.tsv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
OUT = DATA / "feature_selection_ml_results.tsv"

MATRICES = {
    "mRNA": DATA / "external" / "xena" / "HiSeqV2",
    "CNV": DATA / "external" / "xena" / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
    "miRNA": DATA / "external" / "xena" / "miRNA_HiSeq_gene",
}

RANDOM_STATE = 42
K_FOLDS = 5


def make_preprocess(name: str, n_features: int = 200, n_pca: int = 50) -> Pipeline:
    variance = VarianceThreshold(threshold=0.0)

    if name == "PCA50":
        from sklearn.decomposition import PCA

        return Pipeline(
            [
                ("variance", variance),
                ("scale", StandardScaler()),
                ("extract", PCA(n_components=n_pca, random_state=RANDOM_STATE)),
            ]
        )

    if name == "FClassif200":
        selector = SelectKBest(score_func=f_classif, k=n_features)
    elif name == "MutualInfo200":
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    elif name == "L1_LASSO":
        return Pipeline(
            [
                ("variance", variance),
                ("prefilter", SelectKBest(score_func=f_classif, k=1000)),
                (
                    "select",
                    SelectFromModel(
                        LinearSVC(
                            penalty="l1", dual=False, C=0.1, max_iter=5000, random_state=RANDOM_STATE
                        ),
                        max_features=n_features,
                        threshold=-np.inf,
                    ),
                ),
                ("scale", StandardScaler()),
            ]
        )
    elif name == "RFImportance":
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            max_features=n_features,
            threshold=-np.inf,
        )
    elif name == "RFE200":
        return Pipeline(
            [
                ("variance", variance),
                ("prefilter", SelectKBest(score_func=f_classif, k=400)),
                (
                    "rfe",
                    RFE(
                        estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                        n_features_to_select=n_features,
                        step=0.5,
                    ),
                ),
                ("scale", StandardScaler()),
            ]
        )
    else:
        raise ValueError(f"Unknown feature method: {name}")

    return Pipeline(
        [
            ("variance", variance),
            ("select", selector),
            ("scale", StandardScaler()),
        ]
    )


def downstream_classifiers() -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    }


def cv_classification(preprocess_name: str, classifier_name: str, classifier, X, y):
    prep = make_preprocess(preprocess_name)
    model = Pipeline([("prep", prep), ("clf", clone(classifier))])
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
    return {
        "accuracy": float(np.mean(scores["test_accuracy"])),
        "macro_f1": float(np.mean(scores["test_f1_macro"])),
        "balanced_accuracy": float(np.mean(scores["test_balanced_accuracy"])),
    }


def cv_binary_auc(preprocess_name: str, X, y):
    prep = make_preprocess(preprocess_name)
    model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1, error_score="raise")
    return float(np.mean(scores["test_roc_auc"]))


def cv_cox_cindex(preprocess_name: str, X, y_time, y_event):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    c_indices: list[float] = []
    for train_idx, test_idx in cv.split(X, y_event):
        prep = make_preprocess(preprocess_name, n_features=50, n_pca=20)
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
    return float(np.mean(c_indices))


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
    methods = ["FClassif200", "L1_LASSO", "PCA50"]
    rows: list[dict] = []

    for omics, path in MATRICES.items():
        if not path.exists():
            continue
        matrix = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
        matrix = matrix.loc[:, ~matrix.columns.duplicated()]

        # PAM50 4-class
        pam_cases = labels.loc[labels["pam50_4class"].notna() & (labels["pam50_4class"] != ""), "case_id"].tolist()
        X, cases = align_matrix(matrix, pam_cases)
        y = labels.set_index("case_id").loc[cases, "pam50_4class"].values
        for method in methods:
            for clf_name, clf in downstream_classifiers().items():
                try:
                    metrics = cv_classification(method, clf_name, clf, X, y)
                except Exception as exc:  # noqa: BLE001 - keep running on singular failures
                    print(f"  skip {omics}/PAM50/{method}/{clf_name}: {exc}")
                    continue
                rows.append(
                    {
                        "omics": omics,
                        "task": "PAM50_4class",
                        "feature_method": method,
                        "model": clf_name,
                        "metric": "accuracy",
                        "value": round(metrics["accuracy"], 4),
                    }
                )
                rows.append(
                    {
                        "omics": omics,
                        "task": "PAM50_4class",
                        "feature_method": method,
                        "model": clf_name,
                        "metric": "macro_f1",
                        "value": round(metrics["macro_f1"], 4),
                    }
                )
                rows.append(
                    {
                        "omics": omics,
                        "task": "PAM50_4class",
                        "feature_method": method,
                        "model": clf_name,
                        "metric": "balanced_accuracy",
                        "value": round(metrics["balanced_accuracy"], 4),
                    }
                )
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
