#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustly fetch METABRIC mRNA z-scores for the selected genes, with local
caching so the process can be resumed after transient network timeouts."""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_DIR = DATA / "external" / "metabric"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROFILE = "brca_metabric_mrna_median_all_sample_Zscores"
SAMPLE_LIST = "brca_metabric_mrna"
CACHE = OUT_DIR / "metabric_selected_long.tsv"
DONE = OUT_DIR / "metabric_done_genes.txt"
MATRIX_OUT = OUT_DIR / "metabric_mrna_zscores_selected.tsv"


def http_json(url: str, method: str = "GET", payload=None):
    data = None
    headers = {"User-Agent": "Mozilla/5.0"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def symbol_to_entrez(symbols: list[str]) -> dict[str, int]:
    mapping = {}
    for i in range(0, len(symbols), 50):
        chunk = symbols[i : i + 50]
        url = "https://www.cbioportal.org/api/genes/fetch?geneIdType=HUGO_GENE_SYMBOL&projection=SUMMARY"
        rows = http_json(url, method="POST", payload=chunk)
        for row in rows:
            if row.get("hugoGeneSymbol") and row.get("entrezGeneId"):
                mapping[row["hugoGeneSymbol"]] = int(row["entrezGeneId"])
    return mapping


def fetch_one(entrez: int) -> list[dict]:
    url = (
        f"https://www.cbioportal.org/api/molecular-profiles/{PROFILE}/molecular-data"
        f"?sampleListId={SAMPLE_LIST}&entrezGeneId={entrez}&projection=SUMMARY"
    )
    last_err = None
    for attempt in range(4):
        try:
            return http_json(url)
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep(2 * (attempt + 1))
    raise last_err


def load_done() -> set[int]:
    if not DONE.exists():
        return set()
    return {int(x) for x in DONE.read_text(encoding="utf-8").splitlines() if x.strip().isdigit()}


def append_cache(entrez: int, records: list[dict]) -> None:
    with CACHE.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for r in records:
            writer.writerow([entrez, r.get("sampleId"), r.get("value")])
    with DONE.open("a", encoding="utf-8") as f:
        f.write(f"{entrez}\n")


def main() -> None:
    genes = set()
    for path in [
        DATA / "final_datasets" / "PAM50" / "mRNA_PAM50_features.tsv",
        DATA / "final_datasets" / "Survival" / "mRNA_Survival_features.tsv",
    ]:
        df = pd.read_csv(path, sep="\t")
        genes.update(df["feature"].astype(str))
    sym2entrez = symbol_to_entrez(sorted(genes))
    entrez2sym = {v: k for k, v in sym2entrez.items()}
    todo = [e for e in sorted(entrez2sym) if e not in load_done()]
    print(f"genes={len(genes)} mapped={len(sym2entrez)} todo={len(todo)}", flush=True)

    if CACHE.exists() and CACHE.stat().st_size == 0:
        CACHE.unlink()
    if not CACHE.exists():
        with CACHE.open("w", encoding="utf-8", newline="") as f:
            f.write("entrez\tsample\tvalue\n")

    done = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fetch_one, e): e for e in todo}
        for fut in as_completed(futures):
            entrez = futures[fut]
            try:
                records = fut.result()
                append_cache(entrez, records)
            except Exception as err:  # noqa: BLE001
                print(f"FAILED {entrez}: {err}", flush=True)
            done += 1
            if done % 25 == 0:
                print(f"fetched {done}/{len(todo)}", flush=True)

    # Build wide matrix from cache.
    long_df = pd.read_csv(CACHE, sep="\t", dtype={"entrez": str})
    long_df["entrez"] = long_df["entrez"].astype(int)
    wide = long_df.pivot_table(index="entrez", columns="sample", values="value", aggfunc="median")
    wide = wide.rename(index=entrez2sym)
    wide.to_csv(MATRIX_OUT, sep="\t")
    print(f"matrix {wide.shape} -> {MATRIX_OUT}")


if __name__ == "__main__":
    main()
