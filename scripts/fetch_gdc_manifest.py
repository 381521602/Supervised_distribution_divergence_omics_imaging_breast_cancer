#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch TCGA-BRCA omics files and clinical labels from the GDC API.

Outputs (written to ../data):
  brca_files.tsv            file-level records for mRNA / miRNA / CNV
  brca_clinical.tsv         case-level clinical and follow-up labels
  brca_unified_manifest.tsv case-level joined manifest + selected omics files
  data_dictionary.md        field descriptions and data-source notes

The script only queries metadata; it does not download expression matrices.
Use download_gdc_files.py for the actual file downloads.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


BASE_URL = "https://api.gdc.cancer.gov"
PROJECT = "TCGA-BRCA"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"

FILE_FIELDS = [
    "file_id",
    "file_name",
    "data_type",
    "data_category",
    "data_format",
    "experimental_strategy",
    "analysis.workflow_type",
    "access",
    "file_size",
    "cases.submitter_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
    "cases.samples.tissue_type",
]

CASE_FIELDS = [
    "submitter_id",
    "disease_type",
    "primary_site",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.age_at_index",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.sex_at_birth",
    "diagnoses.diagnosis_is_primary_disease",
    "diagnoses.age_at_diagnosis",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.primary_diagnosis",
    "diagnoses.morphology",
    "diagnoses.tumor_grade",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.treatments.therapeutic_agents",
    "diagnoses.treatments.treatment_type",
    "diagnoses.treatments.treatment_intent_type",
    "diagnoses.treatments.treatment_or_therapy",
    "follow_ups.days_to_follow_up",
    "follow_ups.days_to_recurrence",
    "follow_ups.progression_or_recurrence",
    "follow_ups.progression_or_recurrence_type",
    "follow_ups.progression_or_recurrence_anatomic_site",
    "follow_ups.disease_response",
]


def get(path: str, tries: int = 6) -> dict:
    """GET a GDC endpoint with retries for transient network/DNS failures."""
    last_error: Exception | None = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(BASE_URL + path, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - retry any network error
            last_error = exc
            wait = 2 * (attempt + 1)
            time.sleep(wait)
    raise RuntimeError(f"GDC request failed after {tries} attempts: {last_error}")


def project_filter() -> str:
    filt = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {"field": "cases.project.project_id", "value": [PROJECT]},
            }
        ],
    }
    return urllib.parse.quote(json.dumps(filt, separators=(",", ":")))


def case_filter() -> str:
    filt = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "project.project_id", "value": [PROJECT]}},
        ],
    }
    return urllib.parse.quote(json.dumps(filt, separators=(",", ":")))


def fetch_files(data_types: list[str]) -> list[dict]:
    filt = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {"field": "cases.project.project_id", "value": [PROJECT]},
            },
            {"op": "in", "content": {"field": "data_type", "value": data_types}},
        ],
    }
    enc = urllib.parse.quote(json.dumps(filt, separators=(",", ":")))
    fields = urllib.parse.quote(",".join(FILE_FIELDS), safe=",")

    hits: list[dict] = []
    page_size = 2000
    offset = 0
    while True:
        response = get(
            f"/files?filters={enc}&fields={fields}&size={page_size}"
            f"&from={offset}&format=JSON"
        )
        batch = response["data"]["hits"]
        hits.extend(batch)
        total = response["data"]["pagination"]["total"]
        offset += len(batch)
        if offset >= total or not batch:
            break
    return hits


def fetch_cases() -> list[dict]:
    enc = case_filter()
    fields = urllib.parse.quote(",".join(CASE_FIELDS), safe=",")
    expand = "demographic,diagnoses,diagnoses.treatments,follow_ups"
    response = get(
        f"/cases?filters={enc}&fields={fields}&expand={expand}&size=2000&format=JSON"
    )
    return response["data"]["hits"]


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _unique_joined(values) -> str:
    seen: list[str] = []
    for value in _as_list(values):
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return ";".join(seen)


def _pick_primary_diagnosis(case: dict) -> dict:
    diagnoses = _as_list(case.get("diagnoses"))
    if not diagnoses:
        return {}
    for diagnosis in diagnoses:
        if diagnosis.get("diagnosis_is_primary_disease") is True:
            return diagnosis
    return diagnoses[0]


def _followup_summary(case: dict, days_to_death) -> dict:
    follow_ups = _as_list(case.get("follow_ups"))
    days_follow = [
        v for v in (_as_list(fu.get("days_to_follow_up")) for fu in follow_ups)
        if v
    ]
    days_follow = [item for sublist in days_follow for item in sublist]
    max_follow = max(days_follow) if days_follow else None

    recurrence = [
        str(fu.get("progression_or_recurrence"))
        for fu in follow_ups
        if fu.get("progression_or_recurrence") is not None
    ]
    recurrence_yes = any(str(v).lower() in {"yes", "true", "1"} for v in recurrence)

    return {
        "n_followup_records": len(follow_ups),
        "last_followup_days": max_follow,
        "recurrence_or_progression": "Yes" if recurrence_yes else ("No" if recurrence else "Unknown"),
        "recurrence_types": _unique_joined(
            [fu.get("progression_or_recurrence_type") for fu in follow_ups]
        ),
        "recurrence_sites": _unique_joined(
            [fu.get("progression_or_recurrence_anatomic_site") for fu in follow_ups]
        ),
        "disease_response": _unique_joined(
            [fu.get("disease_response") for fu in follow_ups]
        ),
    }


def flatten_files(hits: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for hit in hits:
        cases = _as_list(hit.get("cases"))
        for case in cases:
            samples = _as_list(case.get("samples")) or [{}]
            for sample in samples:
                rows.append(
                    {
                        "file_id": hit.get("file_id", ""),
                        "file_name": hit.get("file_name", ""),
                        "data_type": hit.get("data_type", ""),
                        "data_category": hit.get("data_category", ""),
                        "data_format": hit.get("data_format", ""),
                        "experimental_strategy": hit.get("experimental_strategy", ""),
                        "workflow_type": (hit.get("analysis") or {}).get("workflow_type", ""),
                        "access": hit.get("access", ""),
                        "file_size": hit.get("file_size", ""),
                        "case_id": case.get("submitter_id", ""),
                        "sample_id": sample.get("submitter_id", ""),
                        "sample_type": sample.get("sample_type", ""),
                        "tissue_type": sample.get("tissue_type", ""),
                    }
                )
    return rows


def flatten_cases(hits: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for case in hits:
        demographic = case.get("demographic") or {}
        diagnosis = _pick_primary_diagnosis(case)
        treatments = _as_list(diagnosis.get("treatments"))
        days_to_death = demographic.get("days_to_death")
        followup = _followup_summary(case, days_to_death)

        vital_status = demographic.get("vital_status")
        if vital_status == "Dead" and days_to_death is not None:
            os_time = days_to_death
            os_event = 1
        else:
            os_time = followup["last_followup_days"]
            os_event = 0

        rows.append(
            {
                "case_id": case.get("submitter_id", ""),
                "disease_type": case.get("disease_type", ""),
                "primary_site": case.get("primary_site", ""),
                "vital_status": vital_status or "",
                "days_to_death": days_to_death if days_to_death is not None else "",
                "last_followup_days": followup["last_followup_days"]
                if followup["last_followup_days"] is not None
                else "",
                "os_time_days": os_time if os_time is not None else "",
                "os_event": os_event,
                "age_at_index": demographic.get("age_at_index", ""),
                "race": demographic.get("race", ""),
                "ethnicity": demographic.get("ethnicity", ""),
                "sex_at_birth": demographic.get("sex_at_birth", ""),
                "age_at_diagnosis_days": diagnosis.get("age_at_diagnosis", ""),
                "ajcc_pathologic_stage": diagnosis.get("ajcc_pathologic_stage", ""),
                "ajcc_pathologic_t": diagnosis.get("ajcc_pathologic_t", ""),
                "ajcc_pathologic_n": diagnosis.get("ajcc_pathologic_n", ""),
                "ajcc_pathologic_m": diagnosis.get("ajcc_pathologic_m", ""),
                "primary_diagnosis": diagnosis.get("primary_diagnosis", ""),
                "morphology": diagnosis.get("morphology", ""),
                "tumor_grade": diagnosis.get("tumor_grade", ""),
                "n_treatment_records": len(treatments),
                "therapeutic_agents": _unique_joined(
                    [t.get("therapeutic_agents") for t in treatments]
                ),
                "treatment_types": _unique_joined(
                    [t.get("treatment_type") for t in treatments]
                ),
                "treatment_intent_types": _unique_joined(
                    [t.get("treatment_intent_type") for t in treatments]
                ),
                "treatment_or_therapy": _unique_joined(
                    [t.get("treatment_or_therapy") for t in treatments]
                ),
                "n_followup_records": followup["n_followup_records"],
                "recurrence_or_progression": followup["recurrence_or_progression"],
                "recurrence_types": followup["recurrence_types"],
                "recurrence_sites": followup["recurrence_sites"],
                "disease_response": followup["disease_response"],
            }
        )
    return rows


def select_preferred_file(files: list[dict], data_type: str) -> dict | None:
    candidates = [f for f in files if f["data_type"] == data_type]
    if not candidates:
        return None

    # Prefer primary tumor tissue, then any tumor, then normal.
    def tissue_rank(row: dict) -> int:
        if row["sample_type"] == "primary tumor" and row["tissue_type"] == "tumor":
            return 0
        if row["tissue_type"] == "tumor":
            return 1
        if row["sample_type"] == "solid tissue normal":
            return 2
        return 3

    candidates = sorted(candidates, key=tissue_rank)
    return candidates[0]


def build_unified_manifest(
    clinical_rows: list[dict], file_rows: list[dict]
) -> list[dict]:
    files_by_case: dict[str, list[dict]] = {}
    for row in file_rows:
        files_by_case.setdefault(row["case_id"], []).append(row)

    data_types = [
        "Gene Expression Quantification",
        "miRNA Expression Quantification",
        "Gene Level Copy Number",
        "Masked Copy Number Segment",
    ]
    prefix = {
        "Gene Expression Quantification": "mrna",
        "miRNA Expression Quantification": "mirna",
        "Gene Level Copy Number": "cnv_gene",
        "Masked Copy Number Segment": "cnv_seg",
    }

    rows: list[dict] = []
    for clinical in clinical_rows:
        row = dict(clinical)
        case_files = files_by_case.get(clinical["case_id"], [])
        for data_type in data_types:
            key = prefix[data_type]
            selected = select_preferred_file(case_files, data_type)
            row[f"{key}_file_id"] = selected["file_id"] if selected else ""
            row[f"{key}_file_name"] = selected["file_name"] if selected else ""
            row[f"{key}_sample_id"] = selected["sample_id"] if selected else ""
            row[f"{key}_sample_type"] = selected["sample_type"] if selected else ""
            row[f"{key}_workflow_type"] = selected["workflow_type"] if selected else ""
        n_omics = sum(
            1 for dt in data_types if select_preferred_file(case_files, dt) is not None
        )
        row["n_omics_available"] = n_omics
        rows.append(row)
    return rows


def write_tsv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append(
            "\t".join(
                "" if row.get(col) is None else str(row.get(col)).replace("\t", " ")
                for col in columns
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_data_dictionary(path: Path) -> None:
    text = """# TCGA-BRCA 样本清单与标签字段说明

本目录由 `scripts/fetch_gdc_manifest.py` 从 GDC API 实时生成，数据源为
TCGA-BRCA（Breast Invasive Carcinoma）。

## 文件说明

- `brca_files.tsv`：文件级清单，一行一个“文件-病例-样本”记录，覆盖 mRNA、
  miRNA 和 CNV 三类文件，供后续下载脚本使用。
- `brca_clinical.tsv`：病例级临床与随访标签表，一行一个病例。
- `brca_unified_manifest.tsv`：病例级统一主清单，合并了关键临床标签，并给出
  每个病例在三类组学中选出的首选文件/样本。

## 关键字段

### 病例与样本

- `case_id`：TCGA 病例编号，如 `TCGA-XX-XXXX`。
- `sample_id`：TCGA 样本编号（barcode），用于跨组学对齐。
- `sample_type`：`primary tumor` / `solid tissue normal` / `metastatic`。
- `tissue_type`：`tumor` / `normal`。

### 组学文件字段

- `mrna_*`：Gene Expression Quantification（STAR-Counts）。
- `mirna_*`：miRNA Expression Quantification（BCGSC）。
- `cnv_gene_*`：Gene Level Copy Number（ASCAT2/3、ABSOLUTE 等）。
- `cnv_seg_*`：Masked Copy Number Segment。
- `n_omics_available`：该病例可用的组学类型数量。

### 临床与标签

- `vital_status`：Alive / Dead。
- `os_time_days`：总生存时间（天）。Dead 病例取 `days_to_death`，Alive 病例取
  最后一次随访天数。
- `os_event`：1 表示死亡事件，0 表示删失。
- `ajcc_pathologic_stage`、`ajcc_pathologic_t/n/m`：AJCC 病理分期与 T/N/M。
  其中 `m1` 可用于诊断时是否转移的判断。
- `age_at_index`、`race`、`ethnicity`、`sex_at_birth`：人口学信息。
- `primary_diagnosis`、`morphology`、`tumor_grade`：病理类型与分级。
- `therapeutic_agents`、`treatment_types`、`treatment_intent_types`：治疗药物、
  治疗类型和治疗意图（分号分隔，去重）。
- `recurrence_or_progression`、`recurrence_types`、`recurrence_sites`：
  随访中的复发/进展、复发类型（局部/远处）和复发部位。
- `disease_response`：末次疾病状态（如 tumor free / with tumor）。

## 重要提示

1. GDC 统一临床表中**不含 ER/PR/HER2、PAM50 分子分型和多数病理分级**。
   这些字段需从 TCGA legacy 临床文件、cBioPortal PanCancer Atlas 或 UCSC Xena
   的临床矩阵补充，或使用 mRNA 数据 + `genefu` 重算 PAM50。
2. GDC 的 `Gene Level Copy Number` 来自 ASCAT/ABSOLUTE，不是 GISTIC2 阈值化
   结果。申请书要求的 GISTIC2 `-2/-1/0/1/2` 数据建议从 UCSC Xena 或 Firehose
   legacy 获取，见 `fetch_xena_clinical.py`。
3. 本清单只包含元数据，实际表达矩阵/片段文件需运行
   `scripts/download_gdc_files.py` 下载。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching GDC file records ...")
    file_hits = fetch_files(
        [
            "Gene Expression Quantification",
            "miRNA Expression Quantification",
            "Gene Level Copy Number",
            "Copy Number Segment",
            "Masked Copy Number Segment",
        ]
    )
    file_rows = flatten_files(file_hits)
    write_tsv(file_rows, OUT_DIR / "brca_files.tsv")
    print(f"  wrote {len(file_rows)} file rows")

    print("Fetching GDC clinical records ...")
    case_hits = fetch_cases()
    clinical_rows = flatten_cases(case_hits)
    write_tsv(clinical_rows, OUT_DIR / "brca_clinical.tsv")
    print(f"  wrote {len(clinical_rows)} clinical rows")

    print("Building unified manifest ...")
    manifest_rows = build_unified_manifest(clinical_rows, file_rows)
    write_tsv(manifest_rows, OUT_DIR / "brca_unified_manifest.tsv")
    print(f"  wrote {len(manifest_rows)} manifest rows")

    write_data_dictionary(OUT_DIR / "data_dictionary.md")
    print("  wrote data_dictionary.md")


if __name__ == "__main__":
    main()
