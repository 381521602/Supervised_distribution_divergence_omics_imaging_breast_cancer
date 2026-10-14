# -*- coding: utf-8 -*-
"""Regenerate the mRNA functional-category annotation with corrected keyword
rules and GO/Reactome multi-label IDs for key genes."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"E:\组学数据图像化")
GH = ROOT / "github_release"
D = GH / "data"
OUT = ROOT / "data" / "mRNA_functional_category_annotation.tsv"

CATEGORY_NAMES = {
    0: "Development/Epithelium",
    1: "Signaling/Transport",
    2: "Hormone/Metabolism",
    3: "Immune/Inflammation",
    4: "Cell cycle/Proliferation",
    5: "Other",
}

# Extended keyword rules (same six categories as CNV)
RULES = [
    (0, ["KRT", "CDH", "CLDN", "MUC", "ELF", "EPCAM", "FOXA1", "GATA3", "VIM", "ZEB", "SNAI",
         "TWIST", "LAMC", "LAMB", "ITGA", "ITGB", "COL", "FBN", "DCN", "FOXC", "FOXF", "FOXG",
         "TFF", "AGR", "MGP", "SFN", "KRT"]),
    (1, ["SLC", "ABC", "RAB", "GNA", "GNG", "MAPK", "RAF", "RAS", "RHO", "AKT", "PIK3", "PLC",
         "PRK", "WNT", "FZD", "NOTCH", "TGF", "BMP", "SMAD", "ERBB", "EGFR", "FGFR", "IGF",
         "INSR", "GRB", "SOS"]),
    (2, ["ESR", "PGR", "AR", "CYP", "HSD", "LDLR", "APO", "FASN", "SCD", "ACACA", "PPAR", "LEP",
         "INS", "GCG", "THR", "TSHR", "GH"]),
    (3, ["CD3", "CD4", "CD8", "CD19", "CD79", "MS4A1", "CXCL", "CCL", "IL", "TNF", "IFNG", "HLA",
         "B2M", "IGH", "IGL", "TRAC", "PTPRC", "LYZ", "FCGR"]),
    (4, ["CCN", "CDK", "CDC", "BUB", "AURK", "MKI67", "TOP2A", "TP53", "RB1", "E2F", "PCNA", "MCM",
         "PLK", "CHEK", "BRCA", "RAD", "FANC", "WEE", "MYC", "MDM", "MYB", "BIRC", "CCNE", "CCNB"]),
]

# Curated corrections + GO:BP and Reactome IDs for key genes (multi-label)
KEY_ANNOT = {
    "ESR1": (2, "GO:0030520;GO:0045944", "R-HSA-9006931"),
    "PGR": (2, "GO:0030518;GO:0045944", "R-HSA-9006931"),
    "E2F8": (4, "GO:0007049;GO:0051726", "R-HSA-453279"),
    "FOXC1": (0, "GO:0001701;GO:0007507", "R-HSA-1266738"),
    "MYBL2": (4, "GO:0007049;GO:0051301", "R-HSA-69278"),
    "BIRC5": (4, "GO:0051301;GO:0007067", "R-HSA-174048"),
    "MKI67": (4, "GO:0007049;GO:0051301", "R-HSA-1640170"),
    "TFF1": (0, "GO:0008284;GO:0030154", "R-HSA-9006931"),
    "AGR3": (0, "GO:0007165;GO:0030154", "-"),
}


def clean(f):
    return f.split("|")[0].strip().upper()


def assign(f):
    g = clean(f)
    if g in KEY_ANNOT:
        return KEY_ANNOT[g][0]
    for c, keys in RULES:
        if any(k in g for k in keys):
            return c
    return 5


def main():
    rows = []
    for task, path in [("PAM50", D / "final_datasets/PAM50/mRNA_PAM50_features.tsv"),
                       ("Survival", D / "final_datasets/Survival/mRNA_Survival_features.tsv")]:
        feats = pd.read_csv(path, sep="\t")["feature"].astype(str).tolist()
        for f in feats:
            g = clean(f)
            c = assign(f)
            _, go, rea = KEY_ANNOT.get(g, (5, "", ""))
            rows.append({"task": task, "omics": "mRNA", "feature": f, "category_id": c,
                         "category": CATEGORY_NAMES[c], "GO_BP": go, "Reactome": rea})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, sep="\t", index=False)
    print(df.groupby(["task", "category"]).size().to_string())
    print("saved ->", OUT, len(df), "rows")
    # verify key genes
    for g in ["ESR1", "PGR", "E2F8", "FOXC1", "MYBL2", "BIRC5", "MKI67", "TFF1", "AGR3"]:
        m = df[df["feature"].str.contains(g, regex=False)]
        if len(m):
            print(" ", g, "->", m.iloc[0]["category"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
