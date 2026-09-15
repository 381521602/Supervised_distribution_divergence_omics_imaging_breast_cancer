#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pairwise concatenation fusion for mRNA/CNV/miRNA."""

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
OUT = DATA / "pairwise_fusion_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
OMICS = ["mRNA", "CNV", "miRNA"]
PAIRS = [("mRNA", "CNV"), ("mRNA", "miRNA"), ("CNV", "miRNA")]


def load_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def build_pair(task: str, pair: tuple[str, str]):
    a = load_matrix(FEAT_DIR / f"{pair[0]}_{task}_matrix.tsv")
    b = load_matrix(FEAT_DIR / f"{pair[1]}_{task}_matrix.tsv")
    common = a.index.intersection(b.index)
    common = list(common)
    X = np.hstack([a.loc[common].to_numpy(dtype=float), b.loc[common].to_numpy(dtype=float)])
    return X, np.array(common)


def cv_classification(pair_name: str, X, y) -> list[dict]:
    rows: list[dict] = []
    for model_name, clf in [
        ("LogisticRegression", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
    ]:
        model = Pipeline([("scale", StandardScaler()), ("clf", clf)])
        cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1, error_score="raise")
        rows.append({"pair": pair_name, "task": "PAM50_4class", "model": model_name, "metric": "accuracy", "value": round(float(np.mean(scores["test_accuracy"])), 4)})
        rows.append({"pair": pair_name, "task": "PAM50_4class", "model": model_name, "metric": "macro_f1", "value": round(float(np.mean(scores["test_f1_macro"])), 4)})
    return rows


def cv_binary_auc(pair_name: str, X, y) -> list[dict]:
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc"], n_jobs=1, error_score="raise")
    return [{"pair": pair_name, "task": "OS_binary", "model": "LogisticRegression", "metric": "roc_auc", "value": round(float(np.mean(scores["test_roc_auc"])), 4)}]


def cv_cox(pair_name: str, X, y_time, y_event) -> list[dict]:
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
    return [{"pair": pair_name, "task": "OS_Cox", "model": "CoxPH", "metric": "c_index", "value": round(float(np.mean(c_indices)), 4)}]


def main() -> None:
    labels = pd.read_csv(LABELS, sep="\t", dtype={"case_id": str})
    rows: list[dict] = []

    for task in ["PAM50_4class", "OS"]:
        for pair in PAIRS:
            pair_name = f"{pair[0]}+{pair[1]}"
            X, cases = build_pair(task, pair)
            if task == "PAM50_4class":
                y = labels.set_index("case_id").loc[cases, "pam50_4class"].values
                print(f"{task} {pair_name}: {X.shape[0]} x {X.shape[1]}")
                rows.extend(cv_classification(pair_name, X, y))
            else:
                y_event = labels.set_index("case_id").loc[cases, "os_event"].astype(int).values
                y_time = labels.set_index("case_id").loc[cases, "os_time_days"].astype(float).values
                print(f"{task} {pair_name}: {X.shape[0]} x {X.shape[1]}")
                rows.extend(cv_binary_auc(pair_name, X, y_event))
                rows.extend(cv_cox(pair_name, X, y_time, y_event))

    if rows:
        columns = ["pair", "task", "model", "metric", "value"]
        with OUT.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
