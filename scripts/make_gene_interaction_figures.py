from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data/gene_interaction"
OUT = DIR / "fig_gene_interaction_overview.png"


def main():
    ppi = pd.read_csv(DIR / "mrna_consensus_string_ppi.tsv", sep="\t")
    mm = pd.read_csv(DIR / "miRNA_mRNA_negative_correlations.tsv", sep="\t")
    cm = pd.read_csv(DIR / "cnv_mRNA_correlations.tsv", sep="\t")

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    G = nx.Graph()
    for row in ppi.itertuples():
        G.add_edge(row.gene1, row.gene2, weight=float(row.score))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, node_color="#2E74B5")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.55, edge_color="#8EAADB")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="black")
    ax.set_title("A. Consensus-gene STRING PPI", loc="left", fontsize=11, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(mpimg.imread(DIR / "fig_mrna_coexpression_communities.png"))
    ax.set_title("B. mRNA co-expression communities", loc="left", fontsize=11, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    top_mm = mm.sort_values("rho").head(25).iloc[::-1]
    ax.barh(top_mm["feature1"] + " — " + top_mm["feature2"], top_mm["rho"], color="#C0504D")
    ax.set_xlabel("Spearman rho")
    ax.set_title("C. Top miRNA-mRNA negative correlations", loc="left", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=5)
    ax.axvline(0, color="black", lw=0.8)

    ax = fig.add_subplot(gs[1, 1])
    cm = cm.assign(abs_rho=cm["rho"].abs()).sort_values("abs_rho", ascending=False).head(25).iloc[::-1]
    colors = ["#C0504D" if r < 0 else "#2E74B5" for r in cm["rho"]]
    ax.barh(cm["feature1"] + " — " + cm["feature2"], cm["rho"], color=colors)
    ax.set_xlabel("Spearman rho")
    ax.set_title("D. Top CNV-mRNA correlations", loc="left", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=5)
    ax.axvline(0, color="black", lw=0.8)

    fig.suptitle("Gene interaction analysis from single-omics to multi-omics", fontsize=14, fontweight="bold", color="#1F4D78")
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
