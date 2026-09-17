#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure for the stress/adaptive-reprogramming pathway analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "paper_figures" / "fig_stress_pathway.png"

BLUE = "#2E74B5"
DARK = "#1F4D78"
TEAL = "#0E7C86"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


def main():
    df = pd.read_csv(DATA / "stress_pathway_analysis_results.tsv", sep="\t")
    order = ["Hypoxia", "ROS", "OXPHOS", "UPR", "mTORC1", "Glycolysis", "EMT", "DNA repair"]
    df = df.set_index("pathway").loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    ax = axes[0]
    p = df["enrich_p_fisher"].values
    neglog = [-np.log10(x) if x > 0 else 4.0 for x in p]
    colors = [TEAL if x < 0.05 else "#B0B7C3" for x in p]
    ax.barh(range(len(order))[::-1], neglog, color=colors, edgecolor="white", alpha=0.9)
    ax.axvline(-np.log10(0.05), color="#B91C1C", lw=1.2, ls="--")
    ax.text(-np.log10(0.05) + 0.04, 7.3, "p=0.05", color="#B91C1C", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(order))[::-1])
    ax.set_yticklabels(order, fontsize=11, fontweight="bold")
    ax.set_xlabel("-log10(p), Fisher enrichment in top-500 JSD genes", fontsize=10, fontweight="bold")
    ax.set_title("A. Stress-pathway enrichment", loc="left", fontweight="bold", color=DARK, fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.2, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    hr = df["survival_hr"].values
    lo = df["survival_hr_low"].values
    hi = df["survival_hr_high"].values
    err_low = hr - lo
    err_high = hi - hr
    ypos = np.arange(len(order))[::-1]
    ax.errorbar(hr, ypos, xerr=[err_low, err_high], fmt="o", color=BLUE, ecolor="#9AB1CC",
                capsize=3, elinewidth=1.2, ms=5)
    ax.axvline(1.0, color="#444444", lw=1.0, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(order, fontsize=11, fontweight="bold")
    ax.set_xlabel("Univariate Cox hazard ratio (pathway score)", fontsize=10, fontweight="bold")
    ax.set_title("B. Overall-survival association", loc="left", fontweight="bold", color=DARK, fontsize=13)
    ax.set_xlim(0.45, 1.55)
    ax.grid(axis="x", linestyle="--", alpha=0.2, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
