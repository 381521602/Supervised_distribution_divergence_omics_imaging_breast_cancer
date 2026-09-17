#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the sample-level provenance table from the unified master table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
OUT = DATA / "sample_level_provenance.tsv"


def sample_type(sid):
    if not isinstance(sid, str) or len(sid) < 15:
        return ""
    code = sid[13:15]
    return {"01": "Primary Solid Tumor", "06": "Metastatic", "11": "Solid Tissue Normal"}.get(code, code)


def main():
    df = pd.read_csv(LABELS, sep="\t")
    cols = [
        "case_id", "mrna_sample_id", "mirna_sample_id", "cnv_gene_sample_id",
        "n_omics_available", "pam50_4class", "vital_status", "os_time_days", "os_event",
    ]
    prov = df[[c for c in cols if c in df.columns]].copy()
    prov["sample_type"] = prov["mrna_sample_id"].apply(sample_type)
    # order columns
    out_cols = [
        "case_id", "sample_type", "mrna_sample_id", "mirna_sample_id", "cnv_gene_sample_id",
        "n_omics_available", "pam50_4class", "vital_status", "os_time_days", "os_event",
    ]
    prov = prov[[c for c in out_cols if c in prov.columns]]
    prov = prov.sort_values("case_id")
    prov.to_csv(OUT, sep="\t", index=False)
    print(f"rows={len(prov)}, cols={list(prov.columns)}")
    print(f"sample_type counts:\n{prov['sample_type'].value_counts().to_string()}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
