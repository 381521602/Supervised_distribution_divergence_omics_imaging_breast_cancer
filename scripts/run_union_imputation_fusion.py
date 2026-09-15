#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Union-sample fusion with simple missing-omics handling strategies."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "union_imputation_fusion_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
OMICS = ["mRNA", "CNV", "miRNA"]


def load_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def build_union(task: str):
    matrices = {omics: load_matrix(FEAT_DIR / f"{omics}_{task}_matrix.tsv") for omics in OMICS}
    union = matrices[OMICS[0]].index
    for omics in OMICS[1:]:
        union = union.union(matrices[omics].index)
    union = list(union)
    blocks = []
    indicators = []
    for omics in OMICS:
        m = matrices[omics].reindex(union)
        m.columns = [f"{omics}_{col}" for col in m.columns]
        blocks.append(m)
        indicators.append(m.isna().any(axis=1).astype(int).rename(f"{omics}_missing"))
    X = pd.concat(blocks, axis=1)
    indicator_df = pd.concat(indicators, axis=1)
    return X, indicator_df, np.array(union)


def impute(X: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "mean":
        return X.fillna(X.mean())
    if strategy == "zero":
        return X.fillna(0.0)
    raise ValueError(strategy)


def cv_classification(strategy: str, X, y) -> list[dict]:
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1, error_score="raise")
    return [
        {"strategy": strategy, "task": "PAM50_4class", "model": "LogisticRegression", "metric": "accuracy", "value": round(float(np.mean(scores["test_accuracy"])), 4)},
        {"strategy": strategy, "task": "PAM50_4class", "model": "LogisticRegression", "metric": "macro_f1", "value": round(float(np.mean(scores["test_f1_macro"])), 4)},
    ]


def cv_binary_auc(strategy: str, X, y) -> list[dict]:
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc"], n_jobs=1, error_score="raise")
    return [{"strategy": strategy, "task": "OS_binary", "model": "LogisticRegression", "metric": "roc_auc", "value": round(float(np.mean(scores["test_roc_auc"])), 4)}]


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    rows: list[dict] = []

    for task in ["PAM50_4class", "OS"]:
        X, indicators, cases = build_union(task)
        if task == "PAM50_4class":
            y = labels.set_index("case_id").loc[cases, "pam50_4class"].values
        else:
            y = labels.set_index("case_id").loc[cases, "os_event"].astype(int).values

        for strategy in ["mean", "zero"]:
            X_imp = impute(X, strategy)
            X_with_ind = pd.concat([X_imp, indicators], axis=1)
            if task == "PAM50_4class":
                rows.extend(cv_classification(f"{strategy}+indicators", X_with_ind, y))
                rows.extend(cv_classification(strategy, X_imp, y))
            else:
                rows.extend(cv_binary_auc(f"{strategy}+indicators", X_with_ind, y))
                rows.extend(cv_binary_auc(strategy, X_imp, y))

    if rows:
        columns = ["strategy", "task", "model", "metric", "value"]
        with OUT.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
