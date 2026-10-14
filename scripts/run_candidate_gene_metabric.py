# -*- coding: utf-8 -*-
"""Independent candidate-gene validation in METABRIC."""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"E:\组学数据图像化")
sys.path.insert(0, str(ROOT / "scripts"))
from run_metabric_external_validation import load_metabric_clinical

META = ROOT / "data" / "external" / "metabric"
OUT = ROOT / "data" / "candidate_gene_metabric_validation.tsv"

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

    rows = []
    for g in GENES:
        if g not in matrix.index:
            rows.append([g, "NA", "", "", "", "", ""])
            continue
        vals = matrix.loc[g, samples]
        subtype = pam.loc[samples]
        groups = [vals[subtype == s].dropna() for s in ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]]
        means = [float(gp.mean()) if len(gp) else float("nan") for gp in groups]
        try:
            h, p_kw = stats.kruskal(*[gp.values for gp in groups if len(gp) > 0])
        except Exception:
            h, p_kw = float("nan"), float("nan")
        top_subtype = ["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"][int(np.argmax(means))]
        # survival: median-split log-rank approximated by Cox-free univariate (point-biserial AUC on event)
        valid = vals.dropna()
        surv = pd.DataFrame({"x": valid, "t": os_months.loc[valid.index], "e": event.loc[valid.index]}).dropna()
        if len(surv) > 50 and surv["e"].sum() > 0:
            from lifelines import CoxPHFitter
            cph = CoxPHFitter(penalizer=0.1).fit(surv, duration_col="t", event_col="e")
            hr = float(np.exp(cph.params_.iloc[0]))
            p_cox = float(cph.summary["p"].iloc[0])
        else:
            hr, p_cox = float("nan"), float("nan")
        rows.append([g, f"{means[0]:.3f}", f"{means[1]:.3f}", f"{means[2]:.3f}", f"{means[3]:.3f}",
                     f"{p_kw:.3e}", top_subtype, f"{hr:.3f}", f"{p_cox:.3e}"])

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["gene", "LumA_mean", "LumB_mean", "Basal_mean", "HER2_mean", "kruskal_p", "top_subtype", "survival_hr", "cox_p"])
        w.writerows(rows)
    print("wrote", OUT)
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
