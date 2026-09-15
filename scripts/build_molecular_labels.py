#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a clean molecular-subtype / receptor-status table from Xena clinicalMatrix.

Input:
  data/external/xena/BRCA_clinicalMatrix (download via fetch_xena_clinical.py)

Outputs:
  data/brca_molecular_subtype.tsv  case-level PAM50 + ER/PR/HER2 labels
  data/brca_master_table.tsv       GDC unified manifest merged with those labels
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CLINICAL = ROOT / "data" / "external" / "xena" / "BRCA_clinicalMatrix"
MANIFEST = ROOT / "data" / "brca_unified_manifest.tsv"
OUT_SUBTYPE = ROOT / "data" / "brca_molecular_subtype.tsv"
OUT_MASTER = ROOT / "data" / "brca_master_table.tsv"

COLS = [
    "sampleID",
    "PAM50Call_RNAseq",
    "PAM50_mRNA_nature2012",
    "Integrated_Clusters_with_PAM50__nature2012",
    "ER_Status_nature2012",
    "PR_Status_nature2012",
    "HER2_Final_Status_nature2012",
    "breast_carcinoma_estrogen_receptor_status",
    "breast_carcinoma_progesterone_receptor_status",
    "lab_proc_her2_neu_immunohistochemistry_receptor_status",
]


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


def first_value(row: dict, *fields: str) -> str:
    for field in fields:
        value = str(row.get(field, "") or "").strip()
        if value and value.lower() not in {"na", "not available", "[not available]", "unknown"}:
            return value
    return ""


def normalize_status(value: str) -> str:
    value = value.strip().lower()
    if value in {"positive", "pos", "+"}:
        return "Positive"
    if value in {"negative", "neg", "-"}:
        return "Negative"
    if value in {"indeterminate"}:
        return "Indeterminate"
    if value in {"equivocal"}:
        return "Equivocal"
    return value or ""


def normalize_pam50(value: str) -> str:
    mapping = {
        "basal": "Basal-like",
        "basal-like": "Basal-like",
        "her2": "HER2-enriched",
        "her2-enriched": "HER2-enriched",
        "luma": "Luminal A",
        "luminal a": "Luminal A",
        "lumb": "Luminal B",
        "luminal b": "Luminal B",
        "normal": "Normal-like",
        "normal-like": "Normal-like",
    }
    return mapping.get(value.strip().lower(), value)


def build_subtype_rows(rows: list[dict]) -> list[dict]:
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        sample_id = row.get("sampleID", "")
        if not sample_id.startswith("TCGA-"):
            continue
        case_id = sample_id[:12]
        by_case.setdefault(case_id, []).append(row)

    out: list[dict] = []
    for case_id, case_rows in by_case.items():
        # Prefer the primary-tumor aliquot (-01).
        case_rows = sorted(
            case_rows,
            key=lambda r: (0 if r.get("sampleID", "").endswith("-01") else 1),
        )
        row = case_rows[0]
        pam50_raw = first_value(
            row, "PAM50Call_RNAseq", "PAM50_mRNA_nature2012"
        )
        pam50 = normalize_pam50(pam50_raw)
        out.append(
            {
                "case_id": case_id,
                "sample_id": row.get("sampleID", ""),
                "pam50": pam50,
                "pam50call_rnaseq": row.get("PAM50Call_RNAseq", ""),
                "pam50_mrna_nature2012": row.get("PAM50_mRNA_nature2012", ""),
                "integrated_clusters_with_pam50": row.get(
                    "Integrated_Clusters_with_PAM50__nature2012", ""
                ),
                "er_status": normalize_status(
                    first_value(
                        row,
                        "ER_Status_nature2012",
                        "breast_carcinoma_estrogen_receptor_status",
                    )
                ),
                "pr_status": normalize_status(
                    first_value(
                        row,
                        "PR_Status_nature2012",
                        "breast_carcinoma_progesterone_receptor_status",
                    )
                ),
                "her2_status": normalize_status(
                    first_value(
                        row,
                        "HER2_Final_Status_nature2012",
                        "lab_proc_her2_neu_immunohistochemistry_receptor_status",
                    )
                ),
            }
        )
    return out


def merge_manifest(manifest: list[dict], subtypes: list[dict]) -> list[dict]:
    subtype_by_case = {row["case_id"]: row for row in subtypes}
    merged: list[dict] = []
    for row in manifest:
        new = dict(row)
        sub = subtype_by_case.get(row["case_id"], {})
        new["pam50"] = sub.get("pam50", "")
        new["er_status"] = sub.get("er_status", "")
        new["pr_status"] = sub.get("pr_status", "")
        new["her2_status"] = sub.get("her2_status", "")
        new["molecular_sample_id"] = sub.get("sample_id", "")
        merged.append(new)
    return merged


def main() -> None:
    if not CLINICAL.exists():
        raise SystemExit(
            f"Missing {CLINICAL}; run fetch_xena_clinical.py first."
        )

    clinical_rows = read_tsv(CLINICAL)
    subtypes = build_subtype_rows(clinical_rows)
    write_tsv(subtypes, OUT_SUBTYPE)
    print(f"  wrote {len(subtypes)} rows -> {OUT_SUBTYPE}")

    if MANIFEST.exists():
        manifest = read_tsv(MANIFEST)
        merged = merge_manifest(manifest, subtypes)
        write_tsv(merged, OUT_MASTER)
        print(f"  wrote {len(merged)} rows -> {OUT_MASTER}")


if __name__ == "__main__":
    main()
