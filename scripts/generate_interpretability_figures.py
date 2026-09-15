#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate interpretability figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/interpretability"
OUT = DATA / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)


def barh_importance(df, value_col, title, path, color="#2E74B5"):
    top = df.sort_values(value_col, ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.barh(top["gene"], top[value_col], color=color)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_survival():
    df = pd.read_csv(DATA / "survival_key_factors.tsv", sep="\t")
    df = df.dropna(subset=["hr", "p"]).sort_values("p").head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.barh(df["gene"], df["hr"], color="#1F4D78")
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Hazard ratio")
    ax.set_title("Top univariate Cox genes (survival)")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_survival_hr.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_enrichment():
    df = pd.read_csv(DATA / "mrna_pam50_top100_enrichment.tsv", sep="\t")
    df = df.dropna(subset=["p_value"]).sort_values("p_value").head(15).iloc[::-1]
    df["neg_log_p"] = -__import__("numpy").log10(df["p_value"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["term_name"], df["neg_log_p"], color="#D68A2A")
    ax.set_xlabel("-log10(p)")
    ax.set_title("Top enriched pathways")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_enrichment.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    shap = pd.read_csv(DATA / "mrna_pam50_shap_importance.tsv", sep="\t")
    cnn = pd.read_csv(DATA / "mrna_pam50_cnn_saliency_importance.tsv", sep="\t")
    cons = pd.read_csv(DATA / "mrna_pam50_consensus_key_factors.tsv", sep="\t")
    barh_importance(shap, "mean_importance", "Top SHAP importance (mRNA PAM50)", OUT / "fig_shap_top.png")
    barh_importance(cnn, "mean_importance", "Top CNN saliency genes (mRNA PAM50)", OUT / "fig_cnn_top.png", color="#8FAADC")
    cons_top = cons.sort_values("mean_rank").head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.barh(cons_top["gene"], -cons_top["mean_rank"], color="#2F8F5B")
    ax.set_xlabel("Negative mean rank")
    ax.set_title("Consensus key factors (mRNA PAM50)")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_consensus_top.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig_survival()
    fig_enrichment()
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
