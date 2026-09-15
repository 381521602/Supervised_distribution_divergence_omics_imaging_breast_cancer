#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gene functional enrichment/clustering for PAM50 mRNA features."""

from __future__ import annotations

import json
import urllib.request
import ssl
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parent.parent
FEAT=ROOT/"data/final_datasets/PAM50/mRNA_PAM50_features.tsv"
OUT=ROOT/"data/mRNA_PAM50_gprofiler_results.tsv"


def gprofiler(genes):
    payload=json.dumps({"organism":"hsapiens","query":genes,"sources":["GO:BP","KEGG","REAC"],"user_threshold":0.05}).encode("utf-8")
    req=urllib.request.Request("https://biit.cs.ut.ee/gprofiler/api/gost/profile/",data=payload,headers={"Content-Type":"application/json"})
    ctx=ssl._create_unverified_context()
    with urllib.request.urlopen(req,timeout=120,context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    genes=pd.read_csv(FEAT,sep="\t")["feature"].astype(str).tolist()
    result=gprofiler(genes)
    rows=[]
    for terms in result.get("result",[]):
        source=terms.get("source")
        rows.append({
            "source": source,
            "term_id": terms.get("native"),
            "term_name": terms.get("name"),
            "p_value": terms.get("p_value"),
            "term_size": terms.get("term_size"),
            "query_size": terms.get("query_size"),
            "intersection_size": terms.get("intersection_size"),
            "genes": ";".join(
                str(x) for sub in terms.get("intersections",[]) for x in (sub if isinstance(sub,list) else [sub])
            ),
        })
    df=pd.DataFrame(rows).sort_values("p_value")
    df.to_csv(OUT,sep="\t",index=False)
    print(f"Saved -> {OUT}")
    print(df.head(20).to_string(index=False))


if __name__=="__main__":
    main()
