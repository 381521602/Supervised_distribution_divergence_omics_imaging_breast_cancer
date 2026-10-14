#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stream the METABRIC mRNA z-score matrix and keep only the genes needed for
external validation of the TCGA-derived signatures."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
URL = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/master/"
    "public/brca_metabric/data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt"
)
OUT = DATA / "external" / "metabric" / "metabric_mrna_zscores_filtered.tsv"


def load_gene_set() -> set[str]:
    genes = set()
    for path in [
        DATA / "final_datasets" / "PAM50" / "mRNA_PAM50_features.tsv",
        DATA / "final_datasets" / "Survival" / "mRNA_Survival_features.tsv",
    ]:
        df = pd.read_csv(path, sep="\t")
        genes.update(df["feature"].astype(str))
    return genes


def main() -> None:
    gene_set = load_gene_set()
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    kept = {}
    header = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for i, raw in enumerate(resp):
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if i == 0:
                header = line
                continue
            gene = line.split("\t", 1)[0]
            if gene in gene_set:
                kept.setdefault(gene, []).append(line)

    if header is None:
        raise RuntimeError("No header returned")

    # Aggregate duplicate probes by median across the numeric columns.
    out_rows = []
    for gene, lines in kept.items():
        parsed = [line.split("\t") for line in lines]
        # First two columns are Hugo_Symbol and Entrez_Gene_Id.
        if not parsed:
            continue
        n = len(parsed[0])
        numeric = []
        for row in parsed:
            try:
                numeric.append([float(x) for x in row[2:]])
            except ValueError:
                continue
        if not numeric:
            continue
        med = pd.DataFrame(numeric).median(axis=0).tolist()
        out_rows.append([gene, parsed[0][1]] + med)

    with OUT.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in out_rows:
            f.write("\t".join(str(x) for x in row) + "\n")

    print(f"gene_set={len(gene_set)} kept_genes={len(out_rows)} -> {OUT}")


if __name__ == "__main__":
    main()
