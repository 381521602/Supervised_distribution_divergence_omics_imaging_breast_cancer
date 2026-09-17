#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the complete feature -> functional-category annotation mapping for CNV and miRNA."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from generate_cnv_mirna_category_grids import assign_cnv_categories, assign_mirna_categories  # noqa: E402


DATA = ROOT / "data"
CATEGORY_NAMES = {
    0: "Development/Epithelium",
    1: "Signaling/Transport",
    2: "Hormone/Metabolism",
    3: "Immune/Inflammation",
    4: "Cell cycle/Proliferation",
    5: "Other",
}


def main():
    rows = []
    for task in ["PAM50", "Survival"]:
        for omics, kind in [("CNV", "gene"), ("miRNA", "mirna")]:
            order = pd.read_csv(DATA / "images" / task / omics / "order.tsv", sep="\t")
            features = order["feature"].astype(str).tolist()
            cats = assign_cnv_categories(features) if kind == "gene" else assign_mirna_categories(features)
            for f, c in zip(features, cats):
                rows.append({"task": task, "omics": omics, "feature": f, "category_id": c, "category": CATEGORY_NAMES[c]})

    df = pd.DataFrame(rows)
    out = DATA / "omics_functional_category_annotation.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(df.groupby(["task", "omics", "category"]).size().to_string())
    print(f"Wrote -> {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
