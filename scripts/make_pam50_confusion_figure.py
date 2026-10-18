# -*- coding: utf-8 -*-
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CM = ROOT / "data" / "results" / "pam50_mrna_confusion_matrices.tsv"
OUT = ROOT / "data" / "paper_figures" / "fig_confusion_matrix_pam50.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

classes = ["Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]
models = ["LogisticRegression", "FullSizeCNN"]

rows = list(csv.DictReader(CM.open(encoding="utf-8"), delimiter="\t"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, model in zip(axes, models):
    cm = np.array([[int(r[c]) for c in classes] for r in rows if r["model"] == model])
    acc = cm.diagonal().sum() / cm.sum()
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(range(4)); ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"{model}\nAccuracy {acc:.3f}", fontsize=12)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center",
                    fontsize=9, color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("mRNA PAM50 confusion matrices (5-fold CV, leak-free)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
