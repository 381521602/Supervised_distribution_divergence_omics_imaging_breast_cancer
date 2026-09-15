#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary analyses for the scheme-2 masking experiment.

1. Re-enrichment of mRNA 'Other' genes.
2. Overlap between consensus interpretability genes and functional categories.
"""

from __future__ import annotations

import json
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
FEAT = DATA / "final_datasets/PAM50/mRNA_PAM50_features.tsv"
ORIG_ORDER = DATA / "images/PAM50/mRNA/order.npy"
CAT_GRID = DATA / "images/examples/mRNA_category_grid.npy"
CONSENSUS = DATA / "interpretability/mrna_pam50_consensus_key_factors.tsv"
OUT_ENRICH = DATA / "interpretability/mrna_pam50_other_reenrichment.tsv"
OUT_OVERLAP = DATA / "interpretability/mrna_pam50_category_consensus_overlap.tsv"


def gprofiler(genes):
    payload = json.dumps(
        {"organism": "hsapiens", "query": genes, "sources": ["GO:BP", "KEGG", "REAC"], "user_threshold": 0.05}
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))


def gene_categories():
    genes = pd.read_csv(FEAT, sep="\t")["feature"].astype(str).tolist()
    order = np.load(ORIG_ORDER)
    grid = np.load(CAT_GRID)
    positions = spiral_order(20)
    ordered_cat = np.array([grid[pos] for pos in positions[:400]], dtype=int)
    cat = np.zeros(400, dtype=int)
    cat[order] = ordered_cat
    return genes, cat


def main():
    genes, cat = gene_categories()
    other_genes = [g for g, c in zip(genes, cat) if c == 5]
    print(f"Other genes: {len(other_genes)}")

    result = gprofiler(other_genes)
    rows = []
    for term in result.get("result", []):
        genes_hit = []
        for hits in term.get("intersections", []):
            for h in hits:
                genes_hit.append(h)
        rows.append(
            {
                "source": term.get("source"),
                "term_id": term.get("native"),
                "term_name": term.get("name"),
                "p_value": term.get("p_value"),
                "term_size": term.get("term_size"),
                "intersection_size": term.get("intersection_size"),
                "genes": ";".join(genes_hit),
            }
        )
    enrich = pd.DataFrame(rows).sort_values("p_value")
    enrich.to_csv(OUT_ENRICH, sep="\t", index=False)
    print(f"Saved enrichment -> {OUT_ENRICH}")

    consensus = pd.read_csv(CONSENSUS, sep="\t")
    gene_col = "gene" if "gene" in consensus.columns else consensus.columns[0]
    consensus = consensus.sort_values("mean_rank").head(50).copy()
    consensus_genes = set(consensus[gene_col].astype(str))
    rows_overlap = []
    category_names = {
        0: "Development/Epithelium",
        1: "Signaling/Transport",
        2: "Hormone/Metabolism",
        3: "Immune/Inflammation",
        4: "Cell cycle/Proliferation",
        5: "Other",
    }
    for c in sorted(set(cat.tolist())):
        genes_c = [g for g, cc in zip(genes, cat) if cc == c]
        overlap = [g for g in consensus_genes if g in set(genes_c)]
        rows_overlap.append(
            {
                "category": category_names.get(c, str(c)),
                "category_size": len(genes_c),
                "consensus_overlap": len(overlap),
                "overlap_genes": ";".join(overlap),
            }
        )
    overlap_df = pd.DataFrame(rows_overlap)
    overlap_df.to_csv(OUT_OVERLAP, sep="\t", index=False)
    print(f"Saved overlap -> {OUT_OVERLAP}")


if __name__ == "__main__":
    main()
