# -*- coding: utf-8 -*-
"""Download the manifest-selected primary-tumor gene-level CNV files from GDC."""
import csv
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"E:\组学数据图像化")
FILES = ROOT / "github_release" / "data" / "brca_files.tsv"
MAN = ROOT / "github_release" / "data" / "brca_unified_manifest.tsv"
OUT = ROOT / "data" / "gdc_cnv"
DATA_URL = "https://api.gdc.cancer.gov/data/"


def read(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    man = read(MAN)
    want = {r["cnv_gene_file_id"] for r in man if r.get("cnv_gene_file_id")}
    files = [r for r in read(FILES)
             if r["file_id"] in want
             and r.get("data_type") == "Gene Level Copy Number"
             and r.get("sample_type") == "Primary Tumor"]
    by_id = {r["file_id"]: r for r in files}
    print("files to download:", len(by_id))
    OUT.mkdir(parents=True, exist_ok=True)
    done = 0
    for fid, row in by_id.items():
        dest = OUT / (row["sample_id"][:15] + "_" + row["workflow_type"].replace(" ", "_") + ".tsv")
        if dest.exists() and dest.stat().st_size > 0:
            done += 1
            continue
        for attempt in range(5):
            try:
                with urllib.request.urlopen(DATA_URL + fid, timeout=300) as resp:
                    dest.write_bytes(resp.read())
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        done += 1
        if done % 50 == 0:
            print("progress %d/%d" % (done, len(by_id)), flush=True)
    print("DONE downloaded", done, "files to", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
