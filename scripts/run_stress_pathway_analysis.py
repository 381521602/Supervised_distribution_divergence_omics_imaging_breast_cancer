#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stress / adaptive-reprogramming pathway analysis over the FULL mRNA transcriptome.

The 400-feature NSRE subset is nearly disjoint from canonical stress markers, so the
analysis is performed on the full TCGA-BRCA HiSeqV2 matrix. For eight curated pathways
(Hypoxia, ROS, OXPHOS, UPR, mTORC1, Glycolysis, EMT, DNA repair) we report:
  (1) Fisher enrichment among top-NSRE genes (full-transcriptome universe),
  (2) Kruskal-Wallis association of a per-sample mean-z pathway score with PAM50 subtype,
  (3) univariate Cox association of the pathway score with overall survival.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import fisher_exact, kruskal


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from adaptive_nsre import nsre_between_histograms  # noqa: E402


DATA = ROOT / "data"
LABELS = DATA / "brca_labels_modeling_ready.tsv"
HISEQ = DATA / "external/xena/HiSeqV2"
OUT = DATA / "stress_pathway_analysis_results.tsv"
TOP_N = 500
N_BINS = 10

PATHWAYS = {
    "Hypoxia": ["VEGFA", "SLC2A1", "LDHA", "PGK1", "ENO1", "CA9", "BNIP3", "NDRG1",
                "ADM", "ALDOA", "GAPDH", "TPI1", "P4HA1", "EGLN1", "EGLN3", "LOX",
                "PDK1", "ANGPTL4", "HILPDA", "PFKL", "HIF1A", "HK2"],
    "ROS": ["NQO1", "GCLC", "GCLM", "TXN", "TXNRD1", "SOD1", "SOD2", "CAT", "GPX1",
            "GPX4", "PRDX1", "PRDX2", "HMOX1", "NFE2L2", "GSR", "GSTM1", "GSTP1", "CYBA"],
    "OXPHOS": ["NDUFA1", "NDUFB1", "SDHA", "SDHB", "UQCRC1", "COX5A", "COX6A1",
               "ATP5A1", "ATP5B", "COX4I1", "NDUFS1", "CYCS", "ATP5F1", "UQCRFS1",
               "NDUFV1", "SDHD", "COX7A2", "NDUFB5"],
    "UPR": ["HSPA5", "DDIT3", "ATF4", "ATF6", "XBP1", "ERN1", "EIF2AK3", "HSP90B1",
            "DNAJB9", "EDEM1", "PPP1R15A", "HERPUD1", "PDIA4", "PDIA6", "CALR", "DNAJC3",
            "SEC61A1", "GADD34"],
    "mTORC1": ["MTOR", "RPTOR", "RPS6KB1", "EIF4EBP1", "SREBF1", "TFRC", "GAPDH",
               "LDHA", "PGK1", "SLC2A1", "MTHFD2", "PPAT", "ASNS", "PHGDH", "PSAT1",
               "PKM", "RPS6", "EIF4E"],
    "Glycolysis": ["HK2", "PFKL", "PFKP", "ALDOA", "GAPDH", "PGK1", "PGAM1", "ENO1",
                   "PKM", "LDHA", "SLC2A1", "TPI1", "HK1", "GPI", "GAPDHS", "PGAM4", "ALDOC"],
    "EMT": ["VIM", "CDH1", "CDH2", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "FN1",
            "COL1A1", "MMP2", "MMP9", "CTNNB1", "ACTA2", "SERPINE1", "TGFB1", "ITGB1", "KRT19"],
    "DNA repair": ["BRCA1", "BRCA2", "RAD51", "XRCC1", "XRCC5", "XRCC6", "PARP1",
                   "MLH1", "MSH2", "MSH6", "ATM", "ATR", "CHEK1", "CHEK2", "ERCC1",
                   "ERCC2", "FEN1", "LIG3", "MRE11", "RAD50", "RAD51C", "PALB2"],
}


def nsre_scores(X, y):
    classes = np.unique(y)
    scores = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        feature = X[:, j]
        finite = feature[np.isfinite(feature)]
        if finite.size < 10:
            continue
        qs = np.unique(np.quantile(finite, np.linspace(0, 1, N_BINS + 1)))
        if qs.size < 2:
            continue
        bins = np.digitize(feature, qs[1:-1])
        hists = [np.bincount(bins[y == cls], minlength=N_BINS) for cls in classes]
        pair_scores = [
            nsre_between_histograms(hists[a], hists[b])
            for a in range(len(hists))
            for b in range(a + 1, len(hists))
        ]
        scores[j] = float(np.mean(pair_scores)) if pair_scores else 0.0
    return scores


def zscore(expr, genes):
    sub = expr[[g for g in genes if g in expr.columns]]
    if sub.shape[1] == 0:
        return None
    mu = sub.mean(axis=0)
    sd = sub.std(axis=0).replace(0, 1.0)
    return ((sub - mu) / sd).mean(axis=1).to_numpy()


def main():
    labels = pd.read_csv(LABELS, sep="\t")
    labels = labels[["case_id", "mrna_sample_id", "pam50_4class", "os_time_days", "os_event"]].copy()
    labels["prefix"] = labels["mrna_sample_id"].str[:15]
    labels = labels.dropna(subset=["prefix"])

    print("loading HiSeqV2 ...", flush=True)
    expr = pd.read_csv(HISEQ, sep="\t", index_col=0)
    expr.columns = [c[:15] for c in expr.columns]
    expr = expr.T  # samples x genes
    print("expr shape:", expr.shape, flush=True)

    pam_mask = labels["pam50_4class"].isin(["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"])
    pam_labels = labels[pam_mask]
    idx = pam_labels["prefix"].values
    idx = [s for s in idx if s in expr.index]
    X = expr.loc[idx].to_numpy(dtype=float)
    y = pam_labels.set_index("prefix").loc[idx, "pam50_4class"].values
    print(f"PAM50 cohort: {len(idx)} samples, {X.shape[1]} genes", flush=True)

    print("computing NSRE over full transcriptome ...", flush=True)
    scores = nsre_scores(X, y)
    gene_names = np.array(expr.columns)
    order = np.argsort(scores)[::-1]
    top_genes = set(gene_names[order[:TOP_N]].tolist())
    universe = set(gene_names.tolist())

    rows = []
    for name, genes in PATHWAYS.items():
        in_universe = sorted(set(genes) & universe)
        a = len(set(in_universe) & top_genes)
        b = len(in_universe) - a
        c = len(top_genes) - a
        d = len(universe) - a - b - c
        odds, p_fisher = fisher_exact([[a, b], [c, d]], alternative="greater")

        score = zscore(expr.loc[idx], genes)
        if score is not None:
            groups = [score[y == s] for s in np.unique(y)]
            h, p_kw = kruskal(*groups)
        else:
            h, p_kw = np.nan, np.nan

        # survival association over the full matched cohort
        surv_labels = labels.dropna(subset=["os_time_days"]).copy()
        surv_labels = surv_labels[surv_labels["os_event"].isin([0, 1])]
        s_idx = [s for s in surv_labels["prefix"].values if s in expr.index]
        s_score = zscore(expr.loc[s_idx], genes)
        if s_score is not None and len(s_idx) > 10:
            cdf = pd.DataFrame({
                "score": s_score,
                "time": surv_labels.set_index("prefix").loc[s_idx, "os_time_days"].astype(float).values,
                "event": surv_labels.set_index("prefix").loc[s_idx, "os_event"].astype(int).values,
            })
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cdf, duration_col="time", event_col="event")
            hr = float(np.exp(cph.params_["score"]))
            p_cox = float(cph.summary.loc["score", "p"])
            hr_low = float(np.exp(cph.confidence_intervals_.loc["score", "95% lower-bound"]))
            hr_high = float(np.exp(cph.confidence_intervals_.loc["score", "95% upper-bound"]))
        else:
            hr, p_cox = np.nan, np.nan
            hr_low, hr_high = np.nan, np.nan

        rows.append({
            "pathway": name,
            "genes_in_transcriptome": len(in_universe),
            "genes_in_top500": a,
            "enrich_odds_ratio": round(float(odds), 3),
            "enrich_p_fisher": round(float(p_fisher), 6),
            "pam50_p_kruskal": round(float(p_kw), 6) if not np.isnan(p_kw) else "",
            "survival_hr": round(float(hr), 3) if not np.isnan(hr) else "",
            "survival_hr_low": round(float(hr_low), 3) if not np.isnan(hr_low) else "",
            "survival_hr_high": round(float(hr_high), 3) if not np.isnan(hr_high) else "",
            "survival_p_cox": round(float(p_cox), 6) if not np.isnan(p_cox) else "",
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUT, sep="\t", index=False)
    print(result.to_string(index=False))
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
