# -*- coding: utf-8 -*-
"""Redo main-text Figure 5 (gene interaction analysis) with a cleaner layout."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

ROOT = Path(r"E:\组学数据图像化")
DIR = ROOT / "data" / "gene_interaction"
OUT = ROOT / "data" / "paper_figures" / "fig5_gene_interaction_v2.png"

RED = "#C0504D"
BLUE = "#2E74B5"
DARK = "#1F4D78"
GREY = "#8A8A8A"

HUB = {"ESR1", "PGR", "MYBL2", "BIRC5", "E2F8"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


def panel_A(ax, ppi):
    G = nx.Graph()
    for row in ppi.itertuples():
        G.add_edge(row.gene1, row.gene2, weight=float(row.score))
    pos = nx.spring_layout(G, seed=42, k=1.9, iterations=400)
    degree = dict(G.degree(weight="weight"))
    node_colors = [RED if n in HUB else "#9DB6D4" for n in G.nodes()]
    node_sizes = [620 + 220 * degree[n] for n in G.nodes()]
    edge_widths = [0.8 + 2.6 * G[u][v]["weight"] for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.65,
                           edge_color="#A9BBD4", connectionstyle="arc3,rad=0.10")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                           edgecolors=DARK, linewidths=1.4, alpha=0.95)
    # labels offset outward to avoid overlap
    centroid = np.mean([pos[n] for n in G.nodes()], axis=0)
    label_pos = {}
    for n, (x, y) in pos.items():
        v = np.array([x, y]) - centroid
        norm = float(np.hypot(*v)) or 1.0
        label_pos[n] = np.array([x, y]) + v / norm * 0.16
    for n, (x, y) in label_pos.items():
        ax.text(x, y, n, fontsize=10.5, fontweight="bold", ha="center", va="center",
                color="#173A5E",
                bbox=dict(facecolor="white", edgecolor="#C5D4E8", alpha=0.9,
                          pad=0.35, boxstyle="round,pad=0.18"))
    ax.set_title("A. Consensus-gene STRING PPI", loc="left", fontsize=13, fontweight="bold", color=DARK)
    ax.axis("off")


def panel_B(ax, comm, mrna_df):
    genes = mrna_df.columns.tolist()
    top_genes = comm.sort_values(["community", "gene"]).head(40)["gene"].tolist()
    top_genes = [g for g in top_genes if g in genes]
    corr = np.corrcoef(mrna_df[top_genes].to_numpy(dtype=float), rowvar=False)
    Z = linkage(corr, method="average")
    order = leaves_list(Z)
    corr = corr[np.ix_(order, order)]
    genes_ord = [top_genes[i] for i in order]
    comm_map = dict(zip(comm["gene"], comm["community"]))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(np.arange(len(genes_ord)))
    ax.set_xticklabels(genes_ord, rotation=90, fontsize=9, fontweight="bold")
    ax.set_yticks(np.arange(len(genes_ord)))
    ax.set_yticklabels(genes_ord, fontsize=9, fontweight="bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
    ax.set_anchor("W")
    ax.set_title("B. mRNA co-expression (top 40 genes, clustered)", loc="left",
                 fontsize=13, fontweight="bold", color=DARK)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=8)


def panel_C(ax, mm):
    top = mm.sort_values("rho").head(20).iloc[::-1]
    labels = [f"{r.feature1} \u2014 {r.feature2}" for r in top.itertuples()]
    ax.barh(labels, top["rho"], color=RED, edgecolor="none")
    ax.set_xlabel("Spearman \u03c1", fontsize=11, fontweight="bold")
    ax.set_title("C. Top miRNA\u2013mRNA negative correlations", loc="left",
                 fontsize=13, fontweight="bold", color=DARK)
    ax.tick_params(axis="y", labelsize=8, length=0)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.axvline(0, color="black", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_D(ax, cm):
    cm = cm.assign(abs_rho=cm["rho"].abs()).sort_values("abs_rho", ascending=False).head(20).iloc[::-1]
    labels = [f"{r.feature1} \u2014 {r.feature2}" for r in cm.itertuples()]
    labels = [l.split("|")[0] for l in labels]
    colors = [RED if r < 0 else BLUE for r in cm["rho"]]
    ax.barh(labels, cm["rho"], color=colors, edgecolor="none")
    ax.set_xlabel("Spearman \u03c1", fontsize=11, fontweight="bold")
    ax.set_title("D. Top CNV\u2013mRNA correlations", loc="left",
                 fontsize=13, fontweight="bold", color=DARK)
    ax.tick_params(axis="y", labelsize=8, length=0)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.axvline(0, color="black", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    ppi = pd.read_csv(DIR / "mrna_consensus_string_ppi.tsv", sep="\t")
    mm = pd.read_csv(DIR / "miRNA_mRNA_negative_correlations.tsv", sep="\t")
    cm = pd.read_csv(DIR / "cnv_mRNA_correlations.tsv", sep="\t")
    comm = pd.read_csv(DIR / "mrna_coexpression_communities.tsv", sep="\t")
    mrna_df = pd.read_csv(ROOT / "data/final_datasets/PAM50/mRNA_PAM50_final.tsv", sep="\t").drop(columns=["pam50"])

    fig = plt.figure(figsize=(15.5, 11.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.34, wspace=0.30,
                          left=0.06, right=0.96, top=0.95, bottom=0.06)
    panel_A(fig.add_subplot(gs[0, 0]), ppi)
    panel_B(fig.add_subplot(gs[0, 1]), comm, mrna_df)
    panel_C(fig.add_subplot(gs[1, 0]), mm)
    panel_D(fig.add_subplot(gs[1, 1]), cm)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
