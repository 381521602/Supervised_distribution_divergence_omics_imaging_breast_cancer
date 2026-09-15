#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-omics baselines on each pairwise intersection."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
FEAT_DIR = DATA / "selected_features"
OUT = DATA / "single_omics_pairwise_intersection_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
PAIRS = [("mRNA", "CNV"), ("mRNA", "miRNA"), ("CNV", "miRNA")]


def load_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def cv_classification(omics: str, X, y) -> list[dict]:
    rows: list[dict] = []
    for model_name, clf in [
        ("LogisticRegression", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
    ]:
        model = Pipeline([("scale", StandardScaler()), ("clf", clf)])
        cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1, error_score="raise")
        rows.append({"omics": omics, "task": "PAM50_4class", "model": model_name, "metric": "accuracy", "value": round(float(np.mean(scores["test_accuracy"])), 4)})
        rows.append({"omics": omics, "task": "PAM50_4class", "model": model_name, "metric": "macro_f1", "value": round(float(np.mean(scores["test_f1_macro"])), 4)})
    return rows


def cv_binary_auc(omics: str, X, y) -> list[dict]:
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc"], n_jobs=1, error_score="raise")
    return [{"omics": omics, "task": "OS_binary", "model": "LogisticRegression", "metric": "roc_auc", "value": round(float(np.mean(scores["test_roc_auc"])), 4)}]


def cv_cox(omics: str, X, y_time, y_event) -> list[dict]:
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
    return [{"omics": omics, "task": "OS_Cox", "model": "CoxPH", "metric": "c_index", "value": round(float(np.mean(c_indices)), 4)}]


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    rows: list[dict] = []

    for task in ["PAM50_4class", "OS"]:
        matrices = {omics: load_matrix(FEAT_DIR / f"{omics}_{task}_matrix.tsv") for omics in ["mRNA", "CNV", "miRNA"]}
        for pair in PAIRS:
            common = list(matrices[pair[0]].index.intersection(matrices[pair[1]].index))
            pair_name = f"{pair[0]}+{pair[1]}"
            start = len(rows)
            for omics in pair:
                X = matrices[omics].loc[common].to_numpy(dtype=float)
                if task == "PAM50_4class":
                    y = labels.set_index("case_id").loc[common, "pam50_4class"].values
                    rows.extend(cv_classification(omics, X, y))
                else:
                    y_event = labels.set_index("case_id").loc[common, "os_event"].astype(int).values
                    y_time = labels.set_index("case_id").loc[common, "os_time_days"].astype(float).values
                    rows.extend(cv_binary_auc(omics, X, y_event))
                    rows.extend(cv_cox(omics, X, y_time, y_event))
            for row in rows[start:]:
                row["pair_context"] = pair_name

    if rows:
        columns = ["pair_context", "omics", "task", "model", "metric", "value"]
        with OUT.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
