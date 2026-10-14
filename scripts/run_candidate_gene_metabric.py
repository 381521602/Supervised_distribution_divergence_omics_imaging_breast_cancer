# -*- coding: utf-8 -*-
"""Candidate-gene METABRIC validation v2: subtype source, n/events, HR 95% CI,
BH-FDR q-values, and multivariable (age+NPI+subtype) Cox sensitivity."""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_metabric_external_validation import load_metabric_clinical

META = ROOT / "data" / "external" / "metabric"
OUT = ROOT / "data" / "candidate_gene_metabric_validation_v2.tsv"

GENES = ["ESR1", "TFF1", "AGR3", "FOXC1", "MIA", "FABP7", "CCL19", "MS4A1", "PGR", "MYBL2", "BIRC5", "E2F8"]
SUBTYPE_MAP = {"LumA": "Luminal A", "LumB": "Luminal B", "Basal": "Basal-like", "Her2": "HER2-enriched"}


def main():
    matrix = pd.read_csv(META / "metabric_mrna_zscores_selected.tsv", sep="\t", index_col=0)
    clinical = load_metabric_clinical().set_index("PATIENT_ID")
    common = matrix.columns.intersection(clinical.index)
    sub = clinical.loc[common]
    pam = sub["CLAUDIN_SUBTYPE"].map(SUBTYPE_MAP).dropna()
    samples = matrix.columns.intersection(pam.index)

    os_months = pd.to_numeric(sub["OS_MONTHS"], errors="coerce")
    event = sub["OS_STATUS"].astype(str).str.startswith("1").astype(int)
    age = pd.to_numeric(sub["AGE_AT_DIAGNOSIS"], errors="coerce")
    npi = pd.to_numeric(sub["NPI"], errors="coerce")

    rows = []
    univ_p = {}
    for g in GENES:
        if g not in matrix.index:
            continue
        vals = matrix.loc[g, samples]
        subtype = pam.loc[samples]
        groups = [vals[subtype == s].dropna() for s in ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]]
        means = [float(gp.mean()) if len(gp) else float("nan") for gp in groups]
        try:
            _, p_kw = stats.kruskal(*[gp.values for gp in groups if len(gp) > 0])
        except Exception:
            p_kw = float("nan")
        top = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"][int(np.nanargmax(means))] if not all(np.isnan(means)) else "NA"

        surv = pd.DataFrame({
            "x": vals, "t": os_months.loc[vals.index], "e": event.loc[vals.index],
            "age": age.loc[vals.index], "npi": npi.loc[vals.index],
            "subtype": subtype.loc[vals.index],
        }).dropna(subset=["x", "t", "e"])
        n = len(surv)
        ev = int(surv["e"].sum())
        # univariable Cox
        cph = CoxPHFitter(penalizer=0.1).fit(surv[["x", "t", "e"]], duration_col="t", event_col="e")
        hr = float(np.exp(cph.params_.iloc[0]))
        hr_lo = float(np.exp(cph.confidence_intervals_.iloc[0, 0]))
        hr_hi = float(np.exp(cph.confidence_intervals_.iloc[0, 1]))
        p = float(cph.summary["p"].iloc[0])
        univ_p[g] = p
        # multivariable Cox (age + NPI + subtype), complete-case
        m = surv.dropna(subset=["age", "npi"]).copy()
        m = pd.get_dummies(m, columns=["subtype"], drop_first=True)
        multi = ""
        if len(m) > 50 and m["e"].sum() > 0:
            covs = ["x", "age", "npi"] + [c for c in m.columns if c.startswith("subtype_")]
            try:
                cph2 = CoxPHFitter(penalizer=0.1).fit(m[["t", "e"] + covs], duration_col="t", event_col="e")
                mhr = float(np.exp(cph2.params_["x"]))
                mlo = float(np.exp(cph2.confidence_intervals_.loc["x", "95% lower-bound"]))
                mhi = float(np.exp(cph2.confidence_intervals_.loc["x", "95% upper-bound"]))
                mp = float(cph2.summary.loc["x", "p"])
                multi = f"{mhr:.3f} [{mlo:.3f}, {mhi:.3f}] p={mp:.2e}"
            except Exception as ex:
                multi = "NA"
        rows.append([g, n, ev, f"{means[0]:.3f}", f"{means[1]:.3f}", f"{means[2]:.3f}", f"{means[3]:.3f}",
                     f"{p_kw:.2e}", top, f"{hr:.3f}", f"[{hr_lo:.3f}, {hr_hi:.3f}]", f"{p:.2e}", multi])

    # BH-FDR q for univariable Cox p-values
    genes = [r[0] for r in rows]
    pvals = np.array([univ_p[g] for g in genes], dtype=float)
    # manual Benjamini-Hochberg FDR
    m = len(pvals)
    order = np.argsort(pvals)
    qvals = np.empty(m, dtype=float)
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        q = min(prev, pvals[idx] * m / rank)
        qvals[idx] = q
        prev = q
    qmap = dict(zip(genes, qvals))

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["gene", "n", "events", "LumA_mean", "LumB_mean", "Basal_mean", "HER2_mean",
                    "kruskal_p", "top_subtype", "univ_HR", "univ_HR_95CI", "univ_p", "univ_q", "multivariable"])
        for r in rows:
            r.insert(12, f"{qmap[r[0]]:.2e}")
            w.writerow(r)
    print("wrote", OUT)
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
