#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-omics baseline models with classical machine learning.

For mRNA, CNV and miRNA separately:
  1. PAM50 4-class classification (5-fold stratified CV).
  2. Overall-survival event classification (5-fold stratified CV).
  3. Cox proportional-hazards C-index (5-fold CV) as a survival baseline.

Inputs:
  data/external/xena/HiSeqV2                     mRNA gene expression (Xena)
  data/external/xena/miRNA_HiSeq_gene            miRNA expression (Xena)
  data/external/xena/Gistic2_CopyNumber_...      GISTIC2 thresholded CNV
  data/brca_labels_modeling_ready.tsv            labels

Outputs:
  data/baseline_ml_results.tsv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
OUT = DATA / "baseline_ml_results.tsv"

MATRICES = {
    "mRNA": DATA / "external" / "xena" / "HiSeqV2",
    "CNV": DATA / "external" / "xena" / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
    "miRNA": DATA / "external" / "xena" / "miRNA_HiSeq_gene",
}

RANDOM_STATE = 42
K_FOLDS = 5
N_FEATURES_CLS = 200
N_FEATURES_COX = 100


def read_labels() -> pd.DataFrame:
    return pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})


def load_matrix(path: Path) -> pd.DataFrame:
    """Load a Xena matrix: samples in columns, features in rows."""
    matrix = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()]
    return matrix


def sample_to_case(sample_id: str) -> str:
    return sample_id[:12]


def align_matrix(matrix: pd.DataFrame, case_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return X and row case ids aligned to the requested case order."""
    # Build case -> preferred sample column. Prefer primary tumor (-01).
    case_to_columns: dict[str, list[str]] = {}
    for col in matrix.columns:
        case = sample_to_case(col)
        case_to_columns.setdefault(case, []).append(col)

    case_to_best: dict[str, str] = {}
    for case, cols in case_to_columns.items():
        preferred = [c for c in cols if c.endswith("-01")]
        case_to_best[case] = (preferred or cols)[0]

    aligned_cases: list[str] = []
    aligned_cols: list[str] = []
    for case in case_ids:
        if case in case_to_best:
            aligned_cases.append(case)
            aligned_cols.append(case_to_best[case])

    if not aligned_cases:
        raise RuntimeError("No matching samples found between matrix and labels.")

    X = matrix[aligned_cols].T.to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(aligned_cases)


def classifiers() -> dict[str, Pipeline]:
    def make_pipeline(classifier):
        return Pipeline(
            [
                ("variance", VarianceThreshold(threshold=0.0)),
                ("select", SelectKBest(score_func=f_classif, k=N_FEATURES_CLS)),
                ("scale", StandardScaler()),
                ("clf", classifier),
            ]
        )

    return {
        "LogisticRegression": make_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "RandomForest": make_pipeline(RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        "GradientBoosting": make_pipeline(GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        "SVC": make_pipeline(SVC(kernel="linear", random_state=RANDOM_STATE)),
        "KNN": make_pipeline(KNeighborsClassifier(n_neighbors=5)),
    }


def cv_scores(estimator, X: np.ndarray, y: np.ndarray, scoring: list[str]) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    result = cross_validate(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise"
    )
    return {
        key: float(np.mean(result[f"test_{key}"]))
        for key in scoring
    }


def run_classification(omics: str, X: np.ndarray, y: np.ndarray, task: str) -> list[dict]:
    scoring = ["accuracy", "f1_macro", "balanced_accuracy"]
    rows: list[dict] = []
    for name, estimator in classifiers().items():
        scores = cv_scores(estimator, X, y, scoring)
        rows.append(
            {
                "omics": omics,
                "task": task,
                "model": name,
                "metric": "accuracy",
                "value": round(scores["accuracy"], 4),
            }
        )
        rows.append(
            {
                "omics": omics,
                "task": task,
                "model": name,
                "metric": "macro_f1",
                "value": round(scores["f1_macro"], 4),
            }
        )
        rows.append(
            {
                "omics": omics,
                "task": task,
                "model": name,
                "metric": "balanced_accuracy",
                "value": round(scores["balanced_accuracy"], 4),
            }
        )
    return rows


def run_binary_survival(omics: str, X: np.ndarray, y: np.ndarray) -> list[dict]:
    scoring = ["roc_auc", "balanced_accuracy", "f1"]
    rows: list[dict] = []
    for name, estimator in classifiers().items():
        scores = cv_scores(estimator, X, y, scoring)
        rows.append(
            {
                "omics": omics,
                "task": "OS_death_binary",
                "model": name,
                "metric": "roc_auc",
                "value": round(scores["roc_auc"], 4),
            }
        )
        rows.append(
            {
                "omics": omics,
                "task": "OS_death_binary",
                "model": name,
                "metric": "balanced_accuracy",
                "value": round(scores["balanced_accuracy"], 4),
            }
        )
    return rows


def run_cox(omics: str, X: np.ndarray, y_time: np.ndarray, y_event: np.ndarray) -> list[dict]:
    # Global variance filter (unsupervised) keeps the baseline fast and stable.
    variances = np.var(X, axis=0)
    top_idx = np.argsort(variances)[::-1][:N_FEATURES_COX]
    X_top = X[:, top_idx]

    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    c_indices: list[float] = []
    for train_idx, test_idx in cv.split(X_top, y_event):
        train_df = pd.DataFrame(X_top[train_idx])
        train_df["time"] = y_time[train_idx]
        train_df["event"] = y_event[train_idx].astype(int)
        test_df = pd.DataFrame(X_top[test_idx])

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_df, duration_col="time", event_col="event")
        partial_hazard = cph.predict_partial_hazard(test_df)
        c_index = concordance_index(y_time[test_idx], -partial_hazard, y_event[test_idx])
        c_indices.append(float(c_index))

    return [
        {
            "omics": omics,
            "task": "OS_Cox",
            "model": "CoxPH",
            "metric": "c_index",
            "value": round(float(np.mean(c_indices)), 4),
        }
    ]


def main() -> None:
    labels = read_labels()
    all_rows: list[dict] = []

    for omics, path in MATRICES.items():
        if not path.exists():
            print(f"Skip {omics}: missing {path}")
            continue
        print(f"Loading {omics} matrix ...")
        matrix = load_matrix(path)

        # PAM50 4-class
        pam_cases = labels.loc[labels["pam50_4class"].notna() & (labels["pam50_4class"] != ""), "case_id"].tolist()
        if pam_cases:
            X, aligned_cases = align_matrix(matrix, pam_cases)
            y = labels.set_index("case_id").loc[aligned_cases, "pam50_4class"].values
            print(f"  PAM50 n={len(y)}")
            all_rows.extend(run_classification(omics, X, y, "PAM50_4class"))

        # OS death binary
        os_cases = labels.loc[
            labels["os_event"].isin([0, 1]) & labels["os_time_days"].notna(),
            "case_id",
        ].tolist()
        X, aligned_cases = align_matrix(matrix, os_cases)
        y = labels.set_index("case_id").loc[aligned_cases, "os_event"].astype(int).values
        y_time = labels.set_index("case_id").loc[aligned_cases, "os_time_days"].astype(float).values
        print(f"  OS n={len(y)}")
        all_rows.extend(run_binary_survival(omics, X, y))
        all_rows.extend(run_cox(omics, X, y_time, y))

    if all_rows:
        columns = ["omics", "task", "model", "metric", "value"]
        with OUT.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote {len(all_rows)} result rows -> {OUT}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
