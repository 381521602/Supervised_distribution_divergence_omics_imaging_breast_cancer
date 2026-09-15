#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build functional-category grids for CNV and miRNA scheme-2-style images.

For CNV, gene symbols are annotated with g:Profiler GO:BP terms and grouped
into the same six functional categories as mRNA. For miRNA, MIMAT accessions
are mapped to canonical hsa-miR names via miRBase and grouped by curated
miRNA family/function categories.
"""

from __future__ import annotations

import gzip
import json
import re
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import spiral_order  # noqa: E402


DATA = ROOT / "data"
MIRBASE_MATURE = DATA / "external/mirbase_mirna_mature.txt.gz"


def clean_gene(feature: str) -> str:
    return feature.split("|")[0].strip()


def gprofiler_go(genes):
    payload = json.dumps(
        {"organism": "hsapiens", "query": genes, "sources": ["GO:BP"], "user_threshold": 0.05}
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))


def assign_cnv_categories(features):
    rules = [
        (0, ["KRT", "CDH", "CLDN", "MUC", "ELF", "EPCAM", "FOXA1", "GATA3", "VIM", "ZEB", "SNAI", "TWIST", "LAMC", "LAMB", "ITGA", "ITGB", "COL", "FBN", "DCN"]),
        (1, ["SLC", "ABC", "RAB", "GNA", "GNG", "MAPK", "RAF", "RAS", "RHO", "AKT", "PIK3", "PLC", "PRK", "WNT", "FZD", "NOTCH", "TGF", "BMP", "SMAD", "ERBB", "EGFR", "FGFR", "IGF", "INSR", "GRB", "SOS"]),
        (2, ["ESR", "PGR", "AR", "CYP", "HSD", "LDLR", "APO", "FASN", "SCD", "ACACA", "PPAR", "LEP", "INS", "GCG", "THR", "TSHR", "GH"]),
        (3, ["CD3", "CD4", "CD8", "CD19", "CD79", "MS4A1", "CXCL", "CCL", "IL", "TNF", "IFNG", "HLA", "B2M", "IGH", "IGL", "TRAC", "PTPRC", "LYZ", "FCGR"]),
        (4, ["CCN", "CDK", "CDC", "BUB", "AURK", "MKI67", "TOP2A", "TP53", "RB1", "E2F", "PCNA", "MCM", "PLK", "CHEK", "BRCA", "RAD", "FANC", "WEE", "MYC", "MDM"]),
    ]
    cats = []
    for f in features:
        g = clean_gene(f).upper()
        assigned = 5
        for c, keys in rules:
            if any(k in g for k in keys):
                assigned = c
                break
        cats.append(assigned)
    return cats


def load_mimat_name_map():
    mapping = {}
    with gzip.open(MIRBASE_MATURE, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                mimat = parts[3]
                if mimat.startswith("MIMAT"):
                    mapping[mimat] = parts[1]
    return mapping


def mirna_family(name: str):
    m = re.match(r"hsa-(miR|let)-([0-9]+)([a-z]*)", name)
    if not m:
        return name
    kind, num, suffix = m.groups()
    if kind == "let":
        return "let-" + num
    return "miR-" + num


FAMILY_CATEGORY = {
    "let-7": 0,
    "miR-200": 0,
    "miR-141": 0,
    "miR-429": 0,
    "miR-205": 0,
    "miR-199": 0,
    "miR-203": 0,
    "miR-183": 0,
    "miR-96": 0,
    "miR-182": 0,
    "miR-21": 3,
    "miR-146": 3,
    "miR-155": 3,
    "miR-142": 3,
    "miR-181": 3,
    "miR-150": 3,
    "miR-223": 3,
    "miR-124": 3,
    "miR-26": 3,
    "miR-17": 4,
    "miR-18": 4,
    "miR-19": 4,
    "miR-20": 4,
    "miR-25": 4,
    "miR-92": 4,
    "miR-93": 4,
    "miR-106": 4,
    "miR-15": 4,
    "miR-16": 4,
    "miR-195": 4,
    "miR-497": 4,
    "miR-34": 4,
    "miR-449": 4,
    "miR-424": 4,
    "miR-503": 4,
    "miR-221": 4,
    "miR-222": 4,
    "miR-122": 2,
    "miR-33": 2,
    "miR-148": 2,
    "miR-152": 2,
    "miR-27": 2,
    "miR-30": 2,
    "miR-126": 2,
    "miR-192": 2,
    "miR-194": 2,
    "miR-103": 2,
    "miR-107": 2,
    "miR-130": 2,
    "miR-10": 1,
    "miR-29": 1,
    "miR-125": 1,
    "miR-143": 1,
    "miR-145": 1,
    "miR-191": 1,
    "miR-99": 1,
    "miR-100": 1,
    "miR-101": 1,
    "miR-193": 1,
    "miR-204": 1,
    "miR-211": 1,
    "miR-212": 1,
    "miR-132": 1,
}


def assign_mirna_categories(features):
    mapping = load_mimat_name_map()
    cats = []
    for f in features:
        name = mapping.get(f, "")
        family = mirna_family(name)
        cats.append(FAMILY_CATEGORY.get(family, 5))
    return cats


def save_grid(task, omics, features_in_order, categories, size):
    grid = np.full((size, size), 5, dtype=int)
    positions = spiral_order(size)
    for pos, c in zip(positions, categories[: len(features_in_order)]):
        grid[pos] = c
    out = DATA / "images" / task / omics / "category_grid.npy"
    np.save(out, grid)
    print(f"Saved {out} with categories {np.unique(grid, return_counts=True)}")


def main():
    configs = [
        ("PAM50", "CNV", "data/final_datasets/PAM50/CNV_PAM50_final.tsv", 8, "gene"),
        ("Survival", "CNV", "data/final_datasets/Survival/CNV_Survival_final.tsv", 13, "gene"),
        ("PAM50", "miRNA", "data/final_datasets/PAM50/miRNA_PAM50_final.tsv", 25, "mirna"),
        ("Survival", "miRNA", "data/final_datasets/Survival/miRNA_Survival_final.tsv", 25, "mirna"),
    ]
    for task, omics, datafile, size, kind in configs:
        order_tsv = pd.read_csv(DATA / "images" / task / omics / "order.tsv", sep="\t")
        features = order_tsv["feature"].astype(str).tolist()
        categories = assign_cnv_categories(features) if kind == "gene" else assign_mirna_categories(features)
        save_grid(task, omics, features, categories, size)


if __name__ == "__main__":
    main()
