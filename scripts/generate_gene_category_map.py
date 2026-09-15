#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a colored gene-category map in NSRE order."""

import json
import urllib.request
import ssl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
FEAT=ROOT/"data/final_datasets/PAM50/mRNA_PAM50_features.tsv"
ORDER=ROOT/"data/images/PAM50/mRNA/order.npy"
OUT=ROOT/"data/images/examples/mRNA_gene_category_map.png"
import sys
sys.path.insert(0,str(ROOT/"scripts"))
from run_multistream_cnn import spiral_order


CATEGORIES={
    0:"Development/Epithelium",
    1:"Signaling/Transport",
    2:"Hormone/Metabolism",
    3:"Immune/Inflammation",
    4:"Cell cycle/Proliferation",
    5:"Other",
}
COLORS=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#7f7f7f"]


def fetch_gprofiler(genes):
    payload=json.dumps({"organism":"hsapiens","query":genes,"sources":["GO:BP"],"user_threshold":0.05}).encode()
    req=urllib.request.Request("https://biit.cs.ut.ee/gprofiler/api/gost/profile/",data=payload,headers={"Content-Type":"application/json"})
    ctx=ssl._create_unverified_context()
    with urllib.request.urlopen(req,timeout=120,context=ctx) as r:
        return json.loads(r.read().decode())


def assign_category(gene, data):
    for t in data["result"]:
        term_name=t.get("name","").lower()
        # find gene index in query
        idxs=[i for i,g in enumerate(t.get("query_1",[]))] if False else []
        # query field is 'query_1'? skip, use intersections alignment via data['meta']['query_metadata']['queries']['query_1']
    return 5


def main():
    genes=pd.read_csv(FEAT,sep="\t")["feature"].astype(str).tolist()
    data=fetch_gprofiler(genes)
    # g:Profiler returns intersections aligned to query genes; query list is in meta
    query_genes=genes
    cat=[5]*len(genes)
    for t in data["result"]:
        term=t.get("name","").lower()
        if "develop" in term or "epithel" in term or "differentiation" in term:
            c=0
        elif "transport" in term or "signal" in term:
            c=1
        elif "hormone" in term or "metabolic" in term or "metabolism" in term:
            c=2
        elif "immune" in term or "inflamm" in term:
            c=3
        elif "cell cycle" in term or "proliferat" in term:
            c=4
        else:
            c=5
        inter=t.get("intersections",[])
        for i,sublist in enumerate(inter):
            if sublist and i < len(query_genes):
                gene=query_genes[i]
                if gene in genes:
                    idx=genes.index(gene)
                    if cat[idx]==5:
                        cat[idx]=c
    order=np.load(ORDER)
    # order indexes into genes
    cat_map=np.array(cat)[order]
    size=int(np.ceil(np.sqrt(len(order))))
    grid=np.full((size,size),5,dtype=int)
    positions=spiral_order(size)
    for pos,val in zip(positions,cat_map):
        grid[pos]=val
    cmap=matplotlib.colors.ListedColormap(COLORS)
    fig,ax=plt.subplots(figsize=(5,5),dpi=120)
    ax.imshow(grid,cmap=cmap,vmin=0,vmax=5,aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("mRNA gene category map (NSRE order)",fontsize=10)
    # legend
    import matplotlib.patches as mpatches
    patches=[mpatches.Patch(color=COLORS[i],label=CATEGORIES[i]) for i in range(6)]
    ax.legend(handles=patches,bbox_to_anchor=(1.05,1),loc="upper left",fontsize=7)
    fig.tight_layout(); fig.savefig(OUT,bbox_inches="tight"); plt.close(fig)
    np.save(ROOT/"data/images/examples/mRNA_category_grid.npy",grid)
    print(f"Saved -> {OUT}")


if __name__=="__main__":
    main()
