#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wider survival feature-count sensitivity across all three omics."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_entropy_feature_selection import JSDSelector  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
OUT = DATA / "survival_feature_count_all.tsv"
MATRICES = {
    "mRNA": DATA / "external" / "xena" / "HiSeqV2",
    "CNV": DATA / "external" / "xena" / "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
    "miRNA": DATA / "external" / "xena" / "miRNA_HiSeq_gene",
}
RANDOM_STATE = 42
K_FOLDS = 5
K_VALUES = [20, 50, 100, 200, 300, 500]


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


def binary_auc(omics: str, X, y, k: int) -> dict:
    model = Pipeline(
        [
            ("entropy", JSDSelector(prefilter_k=2000, k=k)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc"], n_jobs=1, error_score="raise")
    return {
        "omics": omics,
        "task": "OS_binary",
        "k": k,
        "metric": "roc_auc",
        "value": round(float(np.mean(scores["test_roc_auc"])), 4),
    }


def cox_cindex(omics: str, X, y_time, y_event, k: int) -> dict:
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    c_indices: list[float] = []
    for train_idx, test_idx in cv.split(X, y_event):
        sel = JSDSelector(prefilter_k=2000, k=k)
        sel.fit(X[train_idx], y_event[train_idx])
        X_train = sel.transform(X[train_idx])
        X_test = sel.transform(X[test_idx])
        train_df = pd.DataFrame(X_train)
        train_df["time"] = y_time[train_idx]
        train_df["event"] = y_event[train_idx].astype(int)
        test_df = pd.DataFrame(X_test)
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_df, duration_col="time", event_col="event")
        hazard = cph.predict_partial_hazard(test_df)
        c_indices.append(float(concordance_index(y_time[test_idx], -hazard, y_event[test_idx])))
    return {
        "omics": omics,
        "task": "OS_Cox",
        "k": k,
        "metric": "c_index",
        "value": round(float(np.mean(c_indices)), 4),
    }


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    rows: list[dict] = []

    for omics, path in MATRICES.items():
        if not path.exists():
            continue
        matrix = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
        matrix = matrix.loc[:, ~matrix.columns.duplicated()]
        os_cases = labels.loc[
            labels["os_event"].isin([0, 1]) & labels["os_time_days"].notna(),
            "case_id",
        ].tolist()
        X, cases = align_matrix(matrix, os_cases)
        y_event = labels.set_index("case_id").loc[cases, "os_event"].astype(int).values
        y_time = labels.set_index("case_id").loc[cases, "os_time_days"].astype(float).values

        for k in K_VALUES:
            try:
                rows.append(binary_auc(omics, X, y_event, k))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {omics}/AUC/k={k}: {exc}")
            try:
                rows.append(cox_cindex(omics, X, y_time, y_event, k))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {omics}/Cox/k={k}: {exc}")

    if rows:
        columns = ["omics", "task", "k", "metric", "value"]
        with OUT.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
