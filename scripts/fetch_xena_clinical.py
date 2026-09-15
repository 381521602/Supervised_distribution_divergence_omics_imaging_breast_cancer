#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download TCGA-BRCA supplementary data from UCSC Xena.

Downloads:
  1. GISTIC2 gene-level thresholded CNV (-2/-1/0/1/2)
  2. BRCA clinical matrix (contains PAM50 and ER/PR/HER2 status columns)

Output goes to ../data/external/xena/.
"""

from __future__ import annotations

import gzip
import shutil
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "external" / "xena"
BASE = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"

FILES = {
    "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz": (
        "TCGA.BRCA.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"
    ),
    "BRCA_clinicalMatrix": "TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix",
}


def download(url: str, dest: Path, tries: int = 6) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists  {dest}")
        return dest

    last_error: Exception | None = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                data = response.read()
            dest.write_bytes(data)
            print(f"  done    {dest} ({len(data)} bytes)")
            return dest
        except Exception as exc:  # noqa: BLE001 - retry network errors
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for local_name, remote in FILES.items():
        print(f"Downloading {local_name} ...")
        dest = OUT_DIR / local_name
        download(BASE + remote, dest)

        if local_name.endswith(".gz"):
            unzipped = dest.with_suffix("")
            if not unzipped.exists() or unzipped.stat().st_size == 0:
                with gzip.open(dest, "rb") as src, unzipped.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  unzipped {unzipped}")


if __name__ == "__main__":
    main()
