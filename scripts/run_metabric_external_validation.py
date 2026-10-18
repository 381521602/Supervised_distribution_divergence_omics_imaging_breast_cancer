#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External validation of the TCGA-derived mRNA models on METABRIC (mRNA)."""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "metabric_external_validation_results.tsv"
METABRIC_MATRIX = DATA / "external" / "metabric" / "metabric_mrna_zscores_selected.tsv"
PROFILE = "brca_metabric_mrna_median_all_sample_Zscores"
SAMPLE_LIST = "brca_metabric_mrna"


def http_json(url: str, method: str = "GET", payload=None):
    last_err = None
    for attempt in range(2):
        try:
            data = None
            headers = {"User-Agent": "Mozilla/5.0"}
            if payload is not None:
                data = json.dumps(payload).encode("utf-8")
                headers["Content-Type"] = "application/json"
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep(3 * (attempt + 1))
    raise last_err


def symbol_to_entrez(symbols: list[str]) -> dict[str, int]:
    mapping = {}
    for i in range(0, len(symbols), 50):
        chunk = symbols[i : i + 50]
        url = "https://www.cbioportal.org/api/genes/fetch?geneIdType=HUGO_GENE_SYMBOL&projection=SUMMARY"
        rows = http_json(url, method="POST", payload=chunk)
        for row in rows:
            if row.get("hugoGeneSymbol") and row.get("entrezGeneId"):
                mapping[row["hugoGeneSymbol"]] = int(row["entrezGeneId"])
        print(f"mapped {min(i + 50, len(symbols))}/{len(symbols)} symbols", flush=True)
    return mapping


def fetch_metabric_matrix(entrez_ids: list[int]) -> pd.DataFrame:
    def fetch_one(entrez: int):
        url = (
            f"https://www.cbioportal.org/api/molecular-profiles/{PROFILE}/molecular-data"
            f"?sampleListId={SAMPLE_LIST}&entrezGeneId={entrez}&projection=SUMMARY"
        )
        return entrez, http_json(url)

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for entrez, records in ex.map(fetch_one, entrez_ids):
            rows.extend(records)
            done += 1
            if done % 25 == 0:
                print(f"fetched {done}/{len(entrez_ids)} genes", flush=True)
    records = {}
    for r in rows:
        records.setdefault(r["entrezGeneId"], []).append((r["sampleId"], float(r["value"])))
    df = pd.DataFrame.from_dict(
        {entrez: dict(vals) for entrez, vals in records.items()}, orient="index"
    )
    return df


def load_tcga_pam50() -> tuple[np.ndarray, list[str], np.ndarray]:
    df = pd.read_csv(DATA / "final_datasets" / "PAM50" / "mRNA_PAM50_final.tsv", sep="\t")
    y = df["pam50"].astype(str).values
    genes = [c for c in df.columns if c != "pam50"]
    X = df[genes].to_numpy(dtype=float)
    return y, genes, X


def load_tcga_survival() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA / "final_datasets" / "Survival" / "mRNA_Survival_final.tsv", sep="\t", index_col=0)
    genes = [c for c in df.columns if c not in ("os_event", "os_time_days")]
    X = df[genes].to_numpy(dtype=float)
    y_time = df["os_time_days"].astype(float).values
    y_event = df["os_event"].astype(int).values
    return df, genes, X, y_time, y_event


def load_metabric_clinical() -> pd.DataFrame:
    path = DATA / "external" / "metabric" / "data_clinical_patient.txt"
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    header = None
    for line in lines:
        if line.startswith("PATIENT_ID"):
            header = line.split("\t")
            break
    data = []
    for line in lines[lines.index(line) + 1 :]:
        if line.startswith("#") or not line.strip():
            continue
        data.append(line.split("\t"))
    df = pd.DataFrame(data, columns=header)
    return df


def prepare_metabric_labels(clinical: pd.DataFrame, matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = clinical.set_index("PATIENT_ID")
    common = matrix.columns.intersection(clinical.index)
    matrix = matrix[common]
    sub = clinical.loc[common]

    subtype_map = {
        "LumA": "Luminal A",
        "LumB": "Luminal B",
        "Basal": "Basal-like",
        "Her2": "HER2-enriched",
    }
    pam = sub["CLAUDIN_SUBTYPE"].map(subtype_map)
    pam_valid = pam.dropna()

    os_months = pd.to_numeric(sub["OS_MONTHS"], errors="coerce")
    os_status = sub["OS_STATUS"].astype(str)
    event = os_status.str.startswith("1").astype(int)
    survival = pd.DataFrame({"os_months": os_months, "os_event": event})
    survival = survival.dropna(subset=["os_months"])
    return pam_valid, survival


def fit_predict_pam50(Xtr, ytr, Xte, name):
    if name == "LogisticRegression":
        model = LogisticRegression(max_iter=3000, random_state=42)
        pred = model.fit(Xtr, ytr).predict(Xte)
        return pred, model
    if name == "LinearSVC":
        model = LinearSVC(random_state=42, max_iter=5000)
        pred = model.fit(Xtr, ytr).predict(Xte)
        return pred, model
    if name == "RandomForest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        pred = model.fit(Xtr, ytr).predict(Xte)
        return pred, model
    if name == "MLP":
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        pred = model.fit(Xtr, ytr).predict(Xte)
        return pred, model
    raise ValueError(name)


def main() -> None:
    # TCGA PAM50
    y_pam, pam_genes, X_pam = load_tcga_pam50()
    # TCGA survival
    tcga_sur, sur_genes, X_sur, sur_time, sur_event = load_tcga_survival()

    matrix_symbols = pd.read_csv(METABRIC_MATRIX, sep="\t", index_col=0)

    clinical = load_metabric_clinical()
    pam_valid, survival = prepare_metabric_labels(clinical, matrix_symbols)

    rows = []

    # PAM50 external validation
    pam_common_genes = [g for g in pam_genes if g in matrix_symbols.index]
    X_meta_pam = matrix_symbols.loc[pam_common_genes].T  # samples x genes
    # Align TCGA columns to same order.
    pam_idx = [pam_genes.index(g) for g in pam_common_genes]
    X_pam_train = np.nan_to_num(X_pam[:, pam_idx], nan=0.0)
    meta_samples = X_meta_pam.index.intersection(pam_valid.index)
    X_meta = np.nan_to_num(X_meta_pam.loc[meta_samples].to_numpy(dtype=float), nan=0.0)
    y_meta = pam_valid.loc[meta_samples].values

    scaler = StandardScaler().fit(X_pam_train)
    X_pam_train_s = scaler.transform(X_pam_train)
    meta_scaler = StandardScaler().fit(X_meta)
    X_meta_s = meta_scaler.transform(X_meta)
    for name in ["LogisticRegression", "LinearSVC", "RandomForest", "MLP"]:
        pred, _ = fit_predict_pam50(X_pam_train_s, y_pam, X_meta_s, name)
        rows.append(
            {
                "task": "PAM50",
                "model": name,
                "metric": "accuracy",
                "value": round(float(accuracy_score(y_meta, pred)), 4),
            }
        )
        rows.append(
            {
                "task": "PAM50",
                "model": name,
                "metric": "macro_f1",
                "value": round(float(f1_score(y_meta, pred, average="macro")), 4),
            }
        )
    print(
        "PAM50 external:",
        {
            "tcga_genes": len(pam_genes),
            "metabric_matched": len(pam_common_genes),
            "external_samples": len(meta_samples),
        },
    )

    # Survival external validation
    sur_common_genes = [g for g in sur_genes if g in matrix_symbols.index]
    X_meta_sur = matrix_symbols.loc[sur_common_genes].T
    tcga_sur_X = tcga_sur[sur_common_genes]
    y_sur_event = tcga_sur["os_event"].astype(int).values
    y_sur_time = tcga_sur["os_time_days"].astype(float).values
    X_sur_train = np.nan_to_num(tcga_sur_X.to_numpy(dtype=float), nan=0.0)

    meta_sur_samples = X_meta_sur.index.intersection(survival.index)
    X_meta_sur_mat = np.nan_to_num(X_meta_sur.loc[meta_sur_samples].to_numpy(dtype=float), nan=0.0)
    meta_sur = survival.loc[meta_sur_samples]
    meta_time = meta_sur["os_months"].astype(float).values * 30.44  # months -> days approx
    meta_event = meta_sur["os_event"].astype(int).values

    scaler_sur = StandardScaler().fit(X_sur_train)
    X_sur_train_s = scaler_sur.transform(X_sur_train)
    meta_scaler_sur = StandardScaler().fit(X_meta_sur_mat)
    X_meta_sur_s = meta_scaler_sur.transform(X_meta_sur_mat)

    for name in ["LogisticRegression", "MLP"]:
        model = LogisticRegression(max_iter=3000, random_state=42) if name == "LogisticRegression" else MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
        )
        model.fit(X_sur_train_s, y_sur_event)
        prob = model.predict_proba(X_meta_sur_s)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({"task": "Survival_death_status", "model": name, "metric": "accuracy", "value": round(float(accuracy_score(meta_event, pred)), 4)})
        rows.append({"task": "Survival_death_status", "model": name, "metric": "balanced_accuracy", "value": round(float(balanced_accuracy_score(meta_event, pred)), 4)})
        rows.append({"task": "Survival_death_status", "model": name, "metric": "roc_auc", "value": round(float(roc_auc_score(meta_event, prob)), 4)})
        rows.append({"task": "Survival_death_status", "model": name, "metric": "c_index", "value": round(float(concordance_index(meta_time, -prob, meta_event)), 4)})

    # Risk-level external validation: median OS high/low-risk labels in both cohorts.
    tcga_risk = (sur_time <= np.median(sur_time)).astype(int)
    meta_risk = (meta_time <= np.median(meta_time)).astype(int)
    for name in ["LogisticRegression", "MLP"]:
        model = LogisticRegression(max_iter=3000, random_state=42) if name == "LogisticRegression" else MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
        )
        model.fit(X_sur_train_s, tcga_risk)
        prob = model.predict_proba(X_meta_sur_s)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({"task": "Survival_risk_level", "model": name, "metric": "accuracy", "value": round(float(accuracy_score(meta_risk, pred)), 4)})
        rows.append({"task": "Survival_risk_level", "model": name, "metric": "balanced_accuracy", "value": round(float(balanced_accuracy_score(meta_risk, pred)), 4)})
        rows.append({"task": "Survival_risk_level", "model": name, "metric": "roc_auc", "value": round(float(roc_auc_score(meta_risk, prob)), 4)})
        rows.append({"task": "Survival_risk_level", "model": name, "metric": "c_index", "value": round(float(concordance_index(meta_time, -prob, meta_event)), 4)})

    print(
        "Survival external:",
        {
            "tcga_genes": len(sur_genes),
            "metabric_matched": len(sur_common_genes),
            "external_samples": len(meta_sur_samples),
        },
    )

    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "model", "metric", "value"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
