# -*- coding: utf-8 -*-
import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\组学数据图像化")
SRC = ROOT / "data" / "perturbation_fidelity_results.tsv"
OUT_TSV = ROOT / "data" / "perturbation_fidelity_clean.tsv"
OUT_AUC = ROOT / "data" / "perturbation_fidelity_auc.tsv"
OUT_FIG = ROOT / "data" / "paper_figures" / "fig_perturbation_fidelity.png"

VALID = {
    "LogisticRegression": ["JSD", "SHAP", "Random"],
    "FullSizeCNN": ["JSD", "Saliency", "Random"],
}

df = pd.read_csv(SRC, sep="\t")
df = df.dropna(subset=["mean_accuracy"])
rows = []
auc_rows = []
for model, rankings in VALID.items():
    for r in rankings:
        sub = df[(df["model"] == model) & (df["ranking"] == r)].sort_values("fraction_removed")
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            rows.append([model, r, row["fraction_removed"], row["mean_accuracy"], row["std_accuracy"]])
        auc = float(np.trapz(sub["mean_accuracy"].values, sub["fraction_removed"].values))
        auc_rows.append([model, r, round(auc, 4)])

with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["model", "ranking", "fraction_removed", "mean_accuracy", "std_accuracy"])
    w.writerows(rows)
with OUT_AUC.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["model", "ranking", "deletion_auc"])
    w.writerows(auc_rows)

clean = pd.read_csv(OUT_TSV, sep="\t")
colors = {"JSD": "#C0504D", "SHAP": "#2E74B5", "Saliency": "#0E7C86", "Random": "#8A8A8A"}
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 11, "axes.edgecolor": "#444444", "axes.linewidth": 0.8})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, m, title in [(axes[0], "LogisticRegression", "A. LogisticRegression"),
                     (axes[1], "FullSizeCNN", "B. FullSizeCNN")]:
    for r in VALID[m]:
        d = clean[(clean["model"] == m) & (clean["ranking"] == r)].sort_values("fraction_removed")
        ax.plot(d["fraction_removed"] * 100, d["mean_accuracy"], "-o", ms=3,
                color=colors[r], label=r, lw=1.6)
    ax.set_xlabel("Top features removed (%)", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold", color="#1F4D78", fontsize=12.5)
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0.15, 0.9)
axes[0].legend(fontsize=9.5, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=300)
plt.close(fig)

print("Deletion AUC (lower = better attribution):")
for r in auc_rows:
    print("  ", r)
print("wrote", OUT_TSV, OUT_AUC, OUT_FIG)
