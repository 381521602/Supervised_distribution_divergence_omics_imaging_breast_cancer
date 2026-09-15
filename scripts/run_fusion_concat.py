#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple concatenation fusion of mRNA + CNV + miRNA."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "fusion_concat_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5


def load_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def build_concat(task: str) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    for omics in ["mRNA", "CNV", "miRNA"]:
        path = FEAT_DIR / f"{omics}_{task}_matrix.tsv"
        matrix = load_matrix(path)
        matrices.append(matrix)
    common = matrices[0].index
    for matrix in matrices[1:]:
        common = common.intersection(matrix.index)
    common = list(common)
    parts = [matrix.loc[common].to_numpy(dtype=float) for matrix in matrices]
    X = np.hstack(parts)
    return X, np.array(common)


def classifiers():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVC": SVC(kernel="linear", random_state=RANDOM_STATE),
    }


def cv_classification(X, y, task) -> list[dict]:
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict] = []
    for name, clf in classifiers().items():
        model = Pipeline([("scale", StandardScaler()), ("clf", clf)])
        scoring = ["accuracy", "f1_macro", "balanced_accuracy"]
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
        rows.extend(
            [
                {
                    "task": task,
                    "model": name,
                    "metric": "accuracy",
                    "value": round(float(np.mean(scores["test_accuracy"])), 4),
                },
                {
                    "task": task,
                    "model": name,
                    "metric": "macro_f1",
                    "value": round(float(np.mean(scores["test_f1_macro"])), 4),
                },
                {
                    "task": task,
                    "model": name,
                    "metric": "balanced_accuracy",
                    "value": round(float(np.mean(scores["test_balanced_accuracy"])), 4),
                },
            ]
        )
    return rows


def cv_binary_auc(X, y) -> list[dict]:
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict] = []
    for name, clf in classifiers().items():
        model = Pipeline([("scale", StandardScaler()), ("clf", clf)])
        scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc"], n_jobs=1, error_score="raise")
        rows.append(
            {
                "task": "OS_binary",
                "model": name,
                "metric": "roc_auc",
                "value": round(float(np.mean(scores["test_roc_auc"])), 4),
            }
        )
    return rows


def cv_cox(X, y_time, y_event) -> list[dict]:
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    c_indices: list[float] = []
    for train_idx, test_idx in cv.split(X, y_event):
        variances = np.var(X[train_idx], axis=0)
        top_idx = np.argsort(variances)[::-1][:100]
        X_train = X[train_idx][:, top_idx]
        X_test = X[test_idx][:, top_idx]
        scaler = StandardScaler().fit(X_train)
        train_df = pd.DataFrame(scaler.transform(X_train))
        train_df["time"] = y_time[train_idx]
        train_df["event"] = y_event[train_idx].astype(int)
        test_df = pd.DataFrame(scaler.transform(X_test))
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_df, duration_col="time", event_col="event")
        hazard = cph.predict_partial_hazard(test_df)
        c_indices.append(float(concordance_index(y_time[test_idx], -hazard, y_event[test_idx])))
    return [
        {
            "task": "OS_Cox",
            "model": "CoxPH",
            "metric": "c_index",
            "value": round(float(np.mean(c_indices)), 4),
        }
    ]


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    rows: list[dict] = []

    # PAM50
    X, cases = build_concat("PAM50_4class")
    y = labels.set_index("case_id").loc[cases, "pam50_4class"].values
    print(f"PAM50 fusion samples: {X.shape[0]} x {X.shape[1]}")
    rows.extend(cv_classification(X, y, "PAM50_4class"))

    # OS
    X, cases = build_concat("OS")
    y_event = labels.set_index("case_id").loc[cases, "os_event"].astype(int).values
    y_time = labels.set_index("case_id").loc[cases, "os_time_days"].astype(float).values
    print(f"OS fusion samples: {X.shape[0]} x {X.shape[1]}")
    rows.extend(cv_binary_auc(X, y_event))
    rows.extend(cv_cox(X, y_time, y_event))

    if rows:
        columns = ["task", "model", "metric", "value"]
        with OUT.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
