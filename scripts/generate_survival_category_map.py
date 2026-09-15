#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Survival mRNA gene-category map."""

import json
import urllib.request
import ssl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys


ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"scripts"))
from run_multistream_cnn import spiral_order


FEAT=ROOT/"data/final_datasets/Survival/mRNA_Survival_features.tsv"
ORDER=ROOT/"data/images/Survival/mRNA/order.npy"
OUT=ROOT/"data/images/examples/Survival_mRNA_gene_category_map.png"
GRID=ROOT/"data/images/examples/Survival_mRNA_category_grid.npy"
CATEGORIES={0:"Development/Epithelium",1:"Signaling/Transport",2:"Hormone/Metabolism",3:"Immune/Inflammation",4:"Cell cycle/Proliferation",5:"Other"}
COLORS=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#7f7f7f"]


def fetch(genes):
    payload=json.dumps({"organism":"hsapiens","query":genes,"sources":["GO:BP"],"user_threshold":0.05}).encode()
    req=urllib.request.Request("https://biit.cs.ut.ee/gprofiler/api/gost/profile/",data=payload,headers={"Content-Type":"application/json"})
    ctx=ssl._create_unverified_context()
    with urllib.request.urlopen(req,timeout=120,context=ctx) as r:
        return json.loads(r.read().decode())


def main():
    genes=pd.read_csv(FEAT,sep="\t")["feature"].astype(str).tolist()
    data=fetch(genes)
    cat=[5]*len(genes)
    for t in data["result"]:
        term=t.get("name","").lower()
        if "develop" in term or "epithel" in term or "differentiation" in term: c=0
        elif "transport" in term or "signal" in term: c=1
        elif "hormone" in term or "metabolic" in term or "metabolism" in term: c=2
        elif "immune" in term or "inflamm" in term: c=3
        elif "cell cycle" in term or "proliferat" in term: c=4
        else: c=5
        for i,sublist in enumerate(t.get("intersections",[])):
            if sublist and i<len(genes):
                if cat[i]==5: cat[i]=c
    order=np.load(ORDER); cat_map=np.array(cat)[order]
    size=15
    grid=np.full((size,size),5,dtype=int)
    for pos,val in zip(spiral_order(size),cat_map): grid[pos]=val
    cmap=matplotlib.colors.ListedColormap(COLORS)
    fig,ax=plt.subplots(figsize=(4,4),dpi=120)
    ax.imshow(grid,cmap=cmap,vmin=0,vmax=5,aspect="equal"); ax.axis("off"); ax.set_title("Survival mRNA gene category map",fontsize=9)
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=COLORS[i],label=CATEGORIES[i]) for i in range(6)],bbox_to_anchor=(1.05,1),loc="upper left",fontsize=6)
    fig.tight_layout(); fig.savefig(OUT,bbox_inches="tight"); plt.close(fig)
    np.save(GRID,grid)
    print(f"Saved -> {OUT}")


if __name__=="__main__":
    main()
