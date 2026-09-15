#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate figures for comprehensive multi-omics integration report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "images/comprehensive_report"
OUT.mkdir(parents=True, exist_ok=True)


def read(path):
    return pd.read_csv(DATA / path, sep="\t")


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)


def fig_triple_single():
    df = read("comprehensive_intersection_single_omics_results.tsv")
    omics = ["mRNA", "CNV", "miRNA"]
    def best_pair(o, task, metric):
        r = df[(df.omics == o) & (df.task == task) & (df.metric == metric)].sort_values("mean", ascending=False).iloc[0]
        return float(r["mean"]), float(r["std"])
    pam_acc = np.array([best_pair(o, "PAM50", "accuracy") for o in omics])
    pam_f1 = np.array([best_pair(o, "PAM50", "macro_f1") for o in omics])
    sur_auc = np.array([best_pair(o, "Survival", "roc_auc") for o in omics])
    sur_ci = np.array([best_pair(o, "Survival", "c_index") for o in omics])
    x = np.arange(len(omics)); w = 0.35
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].bar(x - w / 2, pam_acc[:, 0], w, yerr=pam_acc[:, 1], capsize=3, label="Accuracy", color="#2E74B5")
    ax[0].bar(x + w / 2, pam_f1[:, 0], w, yerr=pam_f1[:, 1], capsize=3, label="Macro-F1", color="#7FA8D9")
    ax[0].set_xticks(x, omics); ax[0].set_ylim(0, 1); ax[0].set_title("Best single-omics on triple intersection")
    ax[0].legend(frameon=False); style(ax[0])
    ax[1].bar(x - w / 2, sur_auc[:, 0], w, yerr=sur_auc[:, 1], capsize=3, label="ROC AUC", color="#1F4D78")
    ax[1].bar(x + w / 2, sur_ci[:, 0], w, yerr=sur_ci[:, 1], capsize=3, label="C-index", color="#8FAADC")
    ax[1].set_xticks(x, omics); ax[1].set_ylim(0, 1); ax[1].set_title("Best survival single-omics on triple intersection")
    ax[1].legend(frameon=False); style(ax[1])
    fig.suptitle("Triple-intersection single-omics baselines")
    fig.tight_layout(); fig.savefig(OUT / "fig_triple_single.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_triple_integration():
    df = read("comprehensive_triple_integration_results.tsv")
    schemes = ["Concat_LogisticRegression", "Concat_SVC", "Concat_MLP", "CNN_TripleConcat", "CNN_TripleGated", "CNN_LateAvg"]
    labels = ["Concat LR", "Concat SVC", "Concat MLP", "CNN Concat", "CNN Gated", "CNN LateAvg"]
    def pair(s, task, metric):
        r = df[(df.scheme == s) & (df.task == task) & (df.metric == metric)].iloc[0]
        return float(r["mean"]), float(r["std"])
    pam_acc = np.array([pair(s, "PAM50", "accuracy") for s in schemes])
    pam_f1 = np.array([pair(s, "PAM50", "macro_f1") for s in schemes])
    sur_auc = np.array([pair(s, "Survival", "roc_auc") for s in schemes])
    sur_ci = np.array([pair(s, "Survival", "c_index") for s in schemes])
    x = np.arange(len(schemes)); w = 0.35
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.7))
    ax[0].bar(x - w / 2, pam_acc[:, 0], w, yerr=pam_acc[:, 1], capsize=3, label="Accuracy", color="#2E74B5")
    ax[0].bar(x + w / 2, pam_f1[:, 0], w, yerr=pam_f1[:, 1], capsize=3, label="Macro-F1", color="#7FA8D9")
    ax[0].set_xticks(x, labels, rotation=15, ha="right"); ax[0].set_ylim(0.7, 1); ax[0].legend(frameon=False); style(ax[0])
    ax[1].bar(x - w / 2, sur_auc[:, 0], w, yerr=sur_auc[:, 1], capsize=3, label="ROC AUC", color="#1F4D78")
    ax[1].bar(x + w / 2, sur_ci[:, 0], w, yerr=sur_ci[:, 1], capsize=3, label="C-index", color="#8FAADC")
    ax[1].set_xticks(x, labels, rotation=15, ha="right"); ax[1].set_ylim(0.5, 0.85); ax[1].legend(frameon=False); style(ax[1])
    fig.suptitle("Triple-omics integration")
    fig.tight_layout(); fig.savefig(OUT / "fig_triple_integration.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_pairwise_integration():
    df = read("comprehensive_pairwise_integration_results.tsv")
    pairs = ["mRNA+CNV", "mRNA+miRNA", "CNV+miRNA"]
    best_rows = []
    for p in pairs:
        for task in ["PAM50", "Survival"]:
            sub = df[(df.pair == p) & (df.task == task)]
            best_rows.append(sub.sort_values("mean", ascending=False).iloc[0])
    best = pd.DataFrame(best_rows)
    pam = best[best.task == "PAM50"].set_index("pair")
    sur = best[best.task == "Survival"].set_index("pair")
    pam_vals = [pam.loc[p, ["mean", "std"]].astype(float).values if p in pam.index else [np.nan, np.nan] for p in pairs]
    sur_vals = [sur.loc[p, ["mean", "std"]].astype(float).values if p in sur.index else [np.nan, np.nan] for p in pairs]
    pam_vals = np.array(pam_vals); sur_vals = np.array(sur_vals)
    x = np.arange(len(pairs)); w = 0.35
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].bar(x, pam_vals[:, 0], w, yerr=pam_vals[:, 1], capsize=3, color="#2E74B5"); ax[0].set_title("Best pairwise PAM50 accuracy"); ax[0].set_xticks(x, pairs); ax[0].set_ylim(0.7, 1); style(ax[0])
    ax[1].bar(x, sur_vals[:, 0], w, yerr=sur_vals[:, 1], capsize=3, color="#1F4D78"); ax[1].set_title("Best pairwise Survival ROC AUC"); ax[1].set_xticks(x, pairs); ax[1].set_ylim(0.6, 0.85); style(ax[1])
    fig.suptitle("Best pairwise integration results")
    fig.tight_layout(); fig.savefig(OUT / "fig_pairwise_integration.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_advanced_stacking():
    def get(df, model, task, metric):
        r = df[(df.model == model) & (df.task == task) & (df.metric == metric)].iloc[0]
        return float(r["mean"]), float(r["std"])
    stack = read("final_9model_stacking_results.tsv")
    tr = read("advanced_method3_transformer_results.tsv")
    ds = read("advanced_method2_deepsurv_results.tsv")
    lr = read("advanced_method4_lowrank_bilinear_results.tsv")
    gn = read("advanced_method5_gnn_results.tsv")
    triple = read("comprehensive_triple_integration_results.tsv")
    def adv(name):
        if name == "Transformer":
            return float(tr[tr.task == "PAM50"][tr.metric == "accuracy"]["mean"].iloc[0]), float(tr[tr.task == "PAM50"][tr.metric == "accuracy"]["std"].iloc[0])
        if name == "Low-rank":
            return float(lr[lr.task == "PAM50"][lr.metric == "accuracy"]["mean"].iloc[0]), float(lr[lr.task == "PAM50"][lr.metric == "accuracy"]["std"].iloc[0])
        if name == "GNN":
            return float(gn[gn.task == "PAM50"][gn.metric == "accuracy"]["mean"].iloc[0]), float(gn[gn.task == "PAM50"][gn.metric == "accuracy"]["std"].iloc[0])
        if name == "Stacking":
            return get(stack, "Stacking_9", "PAM50", "accuracy")
        return (0.9270, 0.0217)
    rows = [
        ("Single mRNA ML", adv("single")),
        ("Transformer", adv("Transformer")),
        ("9-model Stacking", adv("Stacking")),
        ("Low-rank", adv("Low-rank")),
        ("GNN", adv("GNN")),
    ]
    rows_sur = [
        ("Triple Concat MLP", (float(triple[(triple.scheme == "Concat_MLP") & (triple.task == "Survival") & (triple.metric == "c_index")]["mean"].iloc[0]), float(triple[(triple.scheme == "Concat_MLP") & (triple.task == "Survival") & (triple.metric == "c_index")]["std"].iloc[0]))),
        ("CNN LateAvg", (float(triple[(triple.scheme == "CNN_LateAvg") & (triple.task == "Survival") & (triple.metric == "c_index")]["mean"].iloc[0]), float(triple[(triple.scheme == "CNN_LateAvg") & (triple.task == "Survival") & (triple.metric == "c_index")]["std"].iloc[0]))),
        ("9-model Stacking", get(stack, "Stacking_9", "Survival", "c_index")),
        ("DeepSurv", (float(ds[ds.metric == "c_index"]["mean"].iloc[0]), float(ds[ds.metric == "c_index"]["std"].iloc[0]))),
        ("Transformer", (float(tr[tr.task == "Survival"][tr.metric == "c_index"]["mean"].iloc[0]), float(tr[tr.task == "Survival"][tr.metric == "c_index"]["std"].iloc[0]))),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    names = [r[0] for r in rows]; vals = [r[1][0] for r in rows]; errs = [r[1][1] for r in rows]
    ax[0].barh(names[::-1], vals[::-1], xerr=errs[::-1], capsize=3, color="#2E74B5"); ax[0].set_xlim(0.6, 1); ax[0].set_title("PAM50 advanced integration"); style(ax[0])
    names2 = [r[0] for r in rows_sur]; vals2 = [r[1][0] for r in rows_sur]; errs2 = [r[1][1] for r in rows_sur]
    ax[1].barh(names2[::-1], vals2[::-1], xerr=errs2[::-1], capsize=3, color="#1F4D78"); ax[1].set_xlim(0.55, 0.85); ax[1].set_title("Survival advanced integration"); style(ax[1])
    fig.suptitle("Advanced methods and stacking summary")
    fig.tight_layout(); fig.savefig(OUT / "fig_advanced_stacking.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def main():
    fig_triple_single()
    fig_triple_integration()
    fig_pairwise_integration()
    fig_advanced_stacking()
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
