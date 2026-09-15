#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download GDC files listed in data/brca_files.tsv.

Open-access files are downloaded through the GDC data endpoint:
    https://api.gdc.cancer.gov/data/<file_id>

By default this downloads only the files selected in the unified manifest
(one preferred file per omics per case). Use --all to download every row in
brca_files.tsv, or --data-type / --sample-type to filter further.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FILES_TSV = ROOT / "data" / "brca_files.tsv"
MANIFEST_TSV = ROOT / "data" / "brca_unified_manifest.tsv"
RAW_DIR = ROOT / "data" / "raw"
DATA_URL = "https://api.gdc.cancer.gov/data/"


def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def selected_file_ids(manifest_rows: list[dict]) -> set[str]:
    ids: set[str] = set()
    prefixes = ["mrna", "mirna", "cnv_gene", "cnv_seg"]
    for row in manifest_rows:
        for prefix in prefixes:
            value = row.get(f"{prefix}_file_id", "")
            if value:
                ids.add(value)
    return ids


def download_one(file_id: str, file_name: str, data_type: str, tries: int = 5) -> Path:
    safe_type = data_type.replace(" ", "_").lower()
    dest_dir = RAW_DIR / safe_type / file_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists  {dest}")
        return dest

    last_error: Exception | None = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(DATA_URL + file_id, timeout=300) as response:
                data = response.read()
            dest.write_bytes(data)
            print(f"  done    {dest} ({len(data)} bytes)")
            return dest
        except Exception as exc:  # noqa: BLE001 - retry network errors
            last_error = exc
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Failed to download {file_id}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="download every file in brca_files.tsv instead of only manifest-selected files",
    )
    parser.add_argument("--data-type", help="only this data_type (e.g. 'Gene Expression Quantification')")
    parser.add_argument("--sample-type", help="only this sample_type (e.g. 'primary tumor')")
    args = parser.parse_args()

    if not FILES_TSV.exists():
        sys.exit(f"Missing {FILES_TSV}; run fetch_gdc_manifest.py first.")

    file_rows = read_tsv(FILES_TSV)

    if args.all:
        chosen = file_rows
    else:
        if not MANIFEST_TSV.exists():
            sys.exit(f"Missing {MANIFEST_TSV}; run fetch_gdc_manifest.py first.")
        wanted_ids = selected_file_ids(read_tsv(MANIFEST_TSV))
        chosen = [row for row in file_rows if row.get("file_id") in wanted_ids]

    if args.data_type:
        chosen = [row for row in chosen if row.get("data_type") == args.data_type]
    if args.sample_type:
        chosen = [row for row in chosen if row.get("sample_type") == args.sample_type]

    # Deduplicate by file_id while preserving one sample record.
    by_id: dict[str, dict] = {}
    for row in chosen:
        by_id.setdefault(row["file_id"], row)
    unique = list(by_id.values())

    print(f"Downloading {len(unique)} files into {RAW_DIR}")
    for row in unique:
        print(
            f"[{row.get('data_type')}] {row.get('file_id')} -> {row.get('file_name')}"
        )
        download_one(
            row["file_id"],
            row.get("file_name", row["file_id"]),
            row.get("data_type", "unknown"),
        )


if __name__ == "__main__":
    main()
