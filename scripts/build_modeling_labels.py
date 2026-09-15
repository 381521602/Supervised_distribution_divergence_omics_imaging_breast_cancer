#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a modeling-ready label table for TCGA-BRCA.

Inputs:
  data/brca_master_table.tsv
  data/external/xena/BRCA_clinicalMatrix

Output:
  data/brca_labels_modeling_ready.tsv

The table cleans and derives the labels used for downstream modeling:
  - PAM50 (full and 4-class)
  - ER / PR / HER2 binary status and numeric 1/0 flags
  - IHC-like subtype and TNBC flag
  - pathologic stage group, early/late stage, node status, M status
  - OS time and event, plus year-level time
  - recurrence / new-tumor-event labels
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "data" / "brca_master_table.tsv"
CLINICAL = ROOT / "data" / "external" / "xena" / "BRCA_clinicalMatrix"
OUT = ROOT / "data" / "brca_labels_modeling_ready.tsv"


def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _clean(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "na", "not available", "[not available]", "unknown", "not reported", "none"}:
        return ""
    return text


def _num(value):
    text = _clean(value)
    if text == "":
        return ""
    try:
        return float(text)
    except ValueError:
        return ""


def _binary_posneg(value: str, extra_missing: set[str] | None = None) -> str:
    """Return Positive/Negative/'' for a receptor-like field."""
    extra_missing = extra_missing or set()
    value = _clean(value).lower()
    if value == "positive":
        return "Positive"
    if value == "negative":
        return "Negative"
    if value in extra_missing or value in {"indeterminate", "equivocal"}:
        return ""
    return ""


def _binary_flag(value: str) -> str:
    """Convert Positive/Negative label to 1/0, empty for unknown."""
    if value == "Positive":
        return "1"
    if value == "Negative":
        return "0"
    return ""


def _pam50_4class(value: str) -> str:
    value = _clean(value)
    if value in {"Luminal A", "Luminal B", "Basal-like", "HER2-enriched"}:
        return value
    return ""


def _stage_group(value: str) -> str:
    value = _clean(value).lower()
    if value.startswith("stage iv"):
        return "IV"
    if value.startswith("stage iii"):
        return "III"
    if value.startswith("stage ii"):
        return "II"
    if value.startswith("stage i"):
        return "I"
    return ""


def _stage_early_late(group: str) -> str:
    if group in {"I", "II"}:
        return "Early"
    if group in {"III", "IV"}:
        return "Late"
    return ""


def _node_status(value: str) -> str:
    value = _clean(value).lower()
    if value.startswith("n0"):
        return "N0"
    if value.startswith(("n1", "n2", "n3")):
        return "N+"
    return ""


def _m_status(value: str) -> str:
    value = _clean(value).lower()
    if value.startswith("m0"):
        return "M0"
    if value.startswith("m1"):
        return "M1"
    if value.startswith("mx"):
        return "MX"
    return ""


def _ihc_subtype(er: str, pr: str, her2: str) -> str:
    if er == "" and pr == "" and her2 == "":
        return ""
    # HR positive if either ER or PR positive; HR negative only when both known negative.
    if er == "" or pr == "" or her2 == "":
        return ""
    hr_pos = er == "Positive" or pr == "Positive"
    her2_pos = her2 == "Positive"
    if hr_pos and not her2_pos:
        return "HR+/HER2-"
    if hr_pos and her2_pos:
        return "HR+/HER2+"
    if not hr_pos and her2_pos:
        return "HR-/HER2+"
    return "TNBC"


def _histology_group(primary_diagnosis: str, morphology: str) -> str:
    text = f"{_clean(primary_diagnosis)} {_clean(morphology)}".lower()
    if "lobular" in text:
        return "ILC"
    if "duct" in text:
        return "IDC"
    return "Other"


def _os_time_years(days) -> str:
    days = _num(days)
    if days == "":
        return ""
    if days <= 0:
        return ""
    return f"{days / 365.25:.3f}"


def _os_time_days(value) -> str:
    days = _num(value)
    if days == "" or days <= 0:
        return ""
    return str(days)


def build_rows(master: list[dict], xena_by_case: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for row in master:
        case_id = row["case_id"]

        er = _binary_posneg(row.get("er_status", ""))
        pr = _binary_posneg(row.get("pr_status", ""))
        her2 = _binary_posneg(row.get("her2_status", ""), extra_missing={"equivocal"})
        pam50 = _clean(row.get("pam50", ""))

        stage_group = _stage_group(row.get("ajcc_pathologic_stage", ""))
        node_status = _node_status(row.get("ajcc_pathologic_n", ""))

        recurrence_gdc = _clean(row.get("recurrence_or_progression", ""))
        if recurrence_gdc.lower() == "yes":
            recurrence_gdc = "Yes"
        elif recurrence_gdc.lower() == "no":
            recurrence_gdc = "No"
        else:
            recurrence_gdc = ""

        xena = xena_by_case.get(case_id, {})
        new_tumor_event = _clean(xena.get("new_tumor_event_after_initial_treatment", ""))
        if new_tumor_event.lower() == "yes":
            new_tumor_event = "Yes"
        elif new_tumor_event.lower() == "no":
            new_tumor_event = "No"
        else:
            new_tumor_event = ""

        rows.append(
            {
                "case_id": case_id,
                "mrna_sample_id": row.get("mrna_sample_id", ""),
                "mirna_sample_id": row.get("mirna_sample_id", ""),
                "cnv_gene_sample_id": row.get("cnv_gene_sample_id", ""),
                "n_omics_available": row.get("n_omics_available", ""),
                "age_at_index": _num(row.get("age_at_index", "")),
                "race": _clean(row.get("race", "")),
                "sex_at_birth": _clean(row.get("sex_at_birth", "")),
                "pam50": pam50,
                "pam50_4class": _pam50_4class(pam50),
                "er_status": er,
                "pr_status": pr,
                "her2_status": her2,
                "er_pos": _binary_flag(er),
                "pr_pos": _binary_flag(pr),
                "her2_pos": _binary_flag(her2),
                "ihc_subtype": _ihc_subtype(er, pr, her2),
                "tnbc": (
                    "1"
                    if er == "Negative" and pr == "Negative" and her2 == "Negative"
                    else ("0" if er != "" and pr != "" and her2 != "" else "")
                ),
                "ajcc_pathologic_stage": _clean(row.get("ajcc_pathologic_stage", "")),
                "stage_group": stage_group,
                "stage_early_late": _stage_early_late(stage_group),
                "node_status": node_status,
                "node_positive": "1" if node_status == "N+" else ("0" if node_status == "N0" else ""),
                "m_status": _m_status(row.get("ajcc_pathologic_m", "")),
                "histology_group": _histology_group(
                    row.get("primary_diagnosis", ""), row.get("morphology", "")
                ),
                "vital_status": _clean(row.get("vital_status", "")),
                "os_time_days": _os_time_days(row.get("os_time_days", "")),
                "os_event": _clean(row.get("os_event", "")),
                "os_time_years": _os_time_years(row.get("os_time_days", "")),
                "recurrence_or_progression": recurrence_gdc,
                "recurrence": (
                    "1" if recurrence_gdc == "Yes" else ("0" if recurrence_gdc == "No" else "")
                ),
                "new_tumor_event": new_tumor_event,
                "new_tumor_event_binary": (
                    "1" if new_tumor_event == "Yes" else ("0" if new_tumor_event == "No" else "")
                ),
                "days_to_new_tumor_event": _num(
                    xena.get("days_to_new_tumor_event_after_initial_treatment", "")
                ),
            }
        )
    return rows


def xena_case_map(clinical_rows: list[dict]) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for row in clinical_rows:
        sample_id = row.get("sampleID", "")
        if not sample_id.startswith("TCGA-"):
            continue
        case_id = sample_id[:12]
        # Prefer primary-tumor aliquot (-01) if multiple rows exist.
        existing = mapping.get(case_id)
        if existing is None or sample_id.endswith("-01"):
            mapping[case_id] = row
    return mapping


def main() -> None:
    if not MASTER.exists():
        raise SystemExit(f"Missing {MASTER}; run fetch_gdc_manifest.py and build_molecular_labels.py first.")
    if not CLINICAL.exists():
        raise SystemExit(f"Missing {CLINICAL}; run fetch_xena_clinical.py first.")

    master = read_tsv(MASTER)
    clinical = read_tsv(CLINICAL)
    xena = xena_case_map(clinical)
    rows = build_rows(master, xena)
    write_tsv(rows, OUT)

    print(f"Wrote {len(rows)} rows -> {OUT}")
    print("Example columns:", ", ".join(rows[0].keys()) if rows else "none")


if __name__ == "__main__":
    main()
