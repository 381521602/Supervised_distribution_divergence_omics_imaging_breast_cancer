from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data/gene_interaction"
OUT = DIR / "fig_gene_interaction_overview.png"


def main():
    ppi = pd.read_csv(DIR / "mrna_consensus_string_ppi.tsv", sep="\t")
    mm = pd.read_csv(DIR / "miRNA_mRNA_negative_correlations.tsv", sep="\t")
    cm = pd.read_csv(DIR / "cnv_mRNA_correlations.tsv", sep="\t")
    comm = pd.read_csv(DIR / "mrna_coexpression_communities.tsv", sep="\t")
    mrna_df = pd.read_csv(ROOT / "data/final_datasets/PAM50/mRNA_PAM50_final.tsv", sep="\t").drop(columns=["pam50"])
    genes = mrna_df.columns.tolist()
    top_genes = comm.sort_values(["community", "gene"]).head(40)["gene"].tolist()
    top_idx = [genes.index(g) for g in top_genes if g in genes]
    top_genes = [genes[i] for i in top_idx]
    corr = np.corrcoef(mrna_df[top_genes].to_numpy(dtype=float), rowvar=False)

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.28, wspace=0.26)

    ax = fig.add_subplot(gs[0, 0])
    G = nx.Graph()
    for row in ppi.itertuples():
        G.add_edge(row.gene1, row.gene2, weight=float(row.score))
    pos = nx.spring_layout(G, seed=42, k=2.2, iterations=200)
    degree = dict(G.degree(weight="weight"))
    node_colors = [degree[n] for n in G.nodes()]
    node_size = [250 + 80 * degree[n] for n in G.nodes()]
    edge_width = [1.0 + 2.0 * G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_colors, cmap="YlOrRd", edgecolors="#173A5E", linewidths=1.2)
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width, alpha=0.75, edge_color="#8EAADB", connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9.5, font_weight="bold", font_color="#173A5E", bbox=dict(facecolor="white", edgecolor="#B7C9E8", alpha=0.9, pad=0.45, boxstyle="round,pad=0.2"))
    ax.set_title("A. Consensus-gene STRING PPI", loc="left", fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_genes)), top_genes, rotation=90, fontsize=9)
    ax.set_yticks(range(len(top_genes)), top_genes, fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_title("B. mRNA co-expression communities (top 40 genes)", loc="left", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 0])
    top_mm = mm.sort_values("rho").head(25).iloc[::-1]
    ax.barh(top_mm["feature1"] + " — " + top_mm["feature2"], top_mm["rho"], color="#C0504D", edgecolor="none")
    ax.set_xlabel("Spearman rho")
    ax.set_title("C. Top miRNA-mRNA negative correlations", loc="left", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.axvline(0, color="black", lw=0.8)

    ax = fig.add_subplot(gs[1, 1])
    cm = cm.assign(abs_rho=cm["rho"].abs()).sort_values("abs_rho", ascending=False).head(25).iloc[::-1]
    colors = ["#C0504D" if r < 0 else "#2E74B5" for r in cm["rho"]]
    ax.barh(cm["feature1"] + " — " + cm["feature2"], cm["rho"], color=colors, edgecolor="none")
    ax.set_xlabel("Spearman rho")
    ax.set_title("D. Top CNV-mRNA correlations", loc="left", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.axvline(0, color="black", lw=0.8)

    fig.suptitle("Gene interaction analysis from single-omics to multi-omics", fontsize=14, fontweight="bold", color="#1F4D78")
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
