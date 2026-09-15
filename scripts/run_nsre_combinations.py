#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Try several two-stage combinations of NSRE with other feature selectors."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_entropy_feature_selection import NSRESelector  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
OUT = DATA / "nsre_combinations_results.tsv"
MATRICES = {
    "mRNA": DATA / "external" / "xena" / "HiSeqV2",
    "CNV": DATA / "external" / "xena" / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
    "miRNA": DATA / "external" / "xena" / "miRNA_HiSeq_gene",
}

RANDOM_STATE = 42
K_FOLDS = 5


def make_selector(name: str, k: int):
    if name == "FClassif":
        return SelectKBest(score_func=f_classif, k=k)
    if name == "NSRE":
        # Variance prefilter is already handled by the outer VarianceThreshold.
        return NSRESelector(prefilter_k=max(k * 10, 1000), k=k)
    if name == "L1":
        return SelectFromModel(
            LinearSVC(penalty="l1", dual=False, C=0.1, max_iter=5000, random_state=RANDOM_STATE),
            max_features=k,
            threshold=-np.inf,
        )
    if name == "RF":
        return SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            max_features=k,
            threshold=-np.inf,
        )
    raise ValueError(name)


def preprocess(method: str, n_features: int = 200) -> Pipeline:
    variance = VarianceThreshold(threshold=0.0)
    sel1, sel2 = method.split("_")
    # Stage 1 keeps a larger candidate pool; stage 2 narrows to final features.
    k1 = max(n_features * 5, 1000)
    k2 = n_features
    return Pipeline(
        [
            ("variance", variance),
            ("stage1", make_selector(sel1, k1)),
            ("stage2", make_selector(sel2, k2)),
            ("scale", StandardScaler()),
        ]
    )


def cv_classification(omics: str, method: str, X, y) -> list[dict]:
    model = Pipeline(
        [
            ("prep", preprocess(method, n_features=200)),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
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
            "model": "LogisticRegression",
            "metric": "accuracy",
            "value": round(float(np.mean(scores["test_accuracy"])), 4),
        },
        {
            "omics": omics,
            "task": "PAM50_4class",
            "feature_method": method,
            "model": "LogisticRegression",
            "metric": "macro_f1",
            "value": round(float(np.mean(scores["test_f1_macro"])), 4),
        },
    ]


def cv_binary_auc(omics: str, method: str, X, y) -> list[dict]:
    model = Pipeline(
        [
            ("prep", preprocess(method, n_features=200)),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc"], n_jobs=1, error_score="raise")
    return [
        {
            "omics": omics,
            "task": "OS_binary",
            "feature_method": method,
            "model": "LogisticRegression",
            "metric": "roc_auc",
            "value": round(float(np.mean(scores["test_roc_auc"])), 4),
        }
    ]


def cv_cox_cindex(omics: str, method: str, X, y_time, y_event) -> list[dict]:
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    c_indices: list[float] = []
    for train_idx, test_idx in cv.split(X, y_event):
        prep = preprocess(method, n_features=50)
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
            "feature_method": method,
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
    methods = ["FClassif_NSRE", "NSRE_FClassif", "L1_NSRE", "NSRE_L1"]
    rows: list[dict] = []

    for omics, path in MATRICES.items():
        if not path.exists():
            continue
        matrix = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
        matrix = matrix.loc[:, ~matrix.columns.duplicated()]

        pam_cases = labels.loc[labels["pam50_4class"].notna() & (labels["pam50_4class"] != ""), "case_id"].tolist()
        X, cases = align_matrix(matrix, pam_cases)
        y = labels.set_index("case_id").loc[cases, "pam50_4class"].values
        for method in methods:
            try:
                rows.extend(cv_classification(omics, method, X, y))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {omics}/PAM50/{method}: {exc}")

        os_cases = labels.loc[
            labels["os_event"].isin([0, 1]) & labels["os_time_days"].notna(),
            "case_id",
        ].tolist()
        X, cases = align_matrix(matrix, os_cases)
        y_event = labels.set_index("case_id").loc[cases, "os_event"].astype(int).values
        y_time = labels.set_index("case_id").loc[cases, "os_time_days"].astype(float).values
        for method in methods:
            try:
                rows.extend(cv_binary_auc(omics, method, X, y_event))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {omics}/OS_binary/{method}: {exc}")
        for method in ["FClassif_NSRE", "NSRE_FClassif", "L1_NSRE"]:
            try:
                rows.extend(cv_cox_cindex(omics, method, X, y_time, y_event))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {omics}/OS_Cox/{method}: {exc}")

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
