#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate clean, error-bar manuscript figures for the JSD imaging paper."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2E74B5"
DARK = "#1F4D78"
ORANGE = "#D97706"
GREEN = "#2F855A"
RED = "#B42318"
PURPLE = "#6B46C1"
TEAL = "#0E7C86"
GRAY = "#6B7280"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.7,
        "figure.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


def read_tsv(name):
    df = pd.read_csv(DATA / name, sep="\t")
    return df


def grouped_bar(
    ax,
    df,
    metric,
    groups,
    models,
    title,
    ylabel,
    palette=None,
    ylim=None,
):
    if palette is None:
        palette = {
            "LogisticRegression": BLUE,
            "RandomForest": GREEN,
            "GradientBoosting": ORANGE,
            "SVC": PURPLE,
            "KNN": GRAY,
            "MLP": TEAL,
            "FullSizeCNN": RED,
            "CoxPH": DARK,
        }
    x = np.arange(len(groups))
    n = len(models)
    width = 0.78 / n
    for i, model in enumerate(models):
        means = []
        errs = []
        for g in groups:
            sub = df[(df["omics"] == g) & (df["model"] == model) & (df["metric"] == metric)]
            if sub.empty:
                means.append(np.nan)
                errs.append(0.0)
            else:
                means.append(float(sub["mean"].iloc[0]))
                errs.append(float(sub["std"].iloc[0]))
        ax.bar(
            x + (i - (n - 1) / 2) * width,
            means,
            width * 0.88,
            yerr=errs,
            capsize=2.2,
            color=palette.get(model, BLUE),
            label=model,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", color=DARK)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_single_omics_pam50():
    df = read_tsv("comprehensive_batch1_single_omics_results.tsv")
    d = df[df["task"] == "PAM50"].copy()
    models = [
        "LogisticRegression",
        "RandomForest",
        "GradientBoosting",
        "SVC",
        "KNN",
        "MLP",
        "FullSizeCNN",
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True)
    grouped_bar(axes[0], d, "accuracy", ["mRNA", "CNV", "miRNA"], models,
                "A. PAM50 classification: Accuracy", "Accuracy", ylim=(0.45, 1.0))
    grouped_bar(axes[1], d, "macro_f1", ["mRNA", "CNV", "miRNA"], models,
                "B. PAM50 classification: Macro-F1", "Macro-F1", ylim=(0.35, 1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / "fig3_pam50_single_omics.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)


def fig_single_omics_survival():
    df = read_tsv("comprehensive_batch1_single_omics_results.tsv")
    d = df[df["task"] == "Survival"].copy()
    models = [
        "LogisticRegression",
        "RandomForest",
        "GradientBoosting",
        "SVC",
        "KNN",
        "MLP",
        "FullSizeCNN",
        "CoxPH",
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.4), sharex=True)
    grouped_bar(axes[0], d, "roc_auc", ["mRNA", "CNV", "miRNA"], models,
                "A. Survival prediction: ROC AUC", "ROC AUC", ylim=(0.45, 0.80))
    grouped_bar(axes[1], d, "c_index", ["mRNA", "CNV", "miRNA"], models,
                "B. Survival prediction: C-index", "C-index", ylim=(0.45, 0.80))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / "fig4_survival_single_omics.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)


def scheme_bar(df, task, metric, title, ylabel, ylim=None):
    d = df[df["task"] == task].copy()
    schemes = [
        "Concat_LogisticRegression",
        "Concat_RandomForest",
        "Concat_GradientBoosting",
        "Concat_SVC",
        "Concat_KNN",
        "Concat_MLP",
        "CNN_TripleConcat",
        "CNN_TripleGated",
        "CNN_LateAvg",
    ]
    labels = [s.replace("Concat_", "").replace("CNN_Triple", "CNN") for s in schemes]
    means, errs = [], []
    for s in schemes:
        sub = d[(d["scheme"] == s) & (d["metric"] == metric)]
        means.append(float(sub["mean"].iloc[0]))
        errs.append(float(sub["std"].iloc[0]))
    x = np.arange(len(schemes))
    return x, labels, means, errs


def fig_triple_integration():
    df = read_tsv("comprehensive_triple_integration_results.tsv")
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.2))
    specs = [
        ("PAM50", "accuracy", "A. PAM50: triple-omics integration Accuracy", "Accuracy", (0.55, 1.0)),
        ("PAM50", "macro_f1", "B. PAM50: triple-omics integration Macro-F1", "Macro-F1", (0.55, 1.0)),
    ]
    x, labels, means, errs = scheme_bar(df, "PAM50", "accuracy", "", "")
    axes[0].bar(x, means, yerr=errs, capsize=2.5, color=BLUE, alpha=0.9, edgecolor="white", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(specs[0][2], loc="left", fontweight="bold", color=DARK)
    axes[0].set_ylim(*specs[0][4])
    axes[0].grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.6)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    x, labels, means, errs = scheme_bar(df, "PAM50", "macro_f1", "", "")
    axes[1].bar(x, means, yerr=errs, capsize=2.5, color=DARK, alpha=0.9, edgecolor="white", linewidth=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title(specs[1][2], loc="left", fontweight="bold", color=DARK)
    axes[1].set_ylim(*specs[1][4])
    axes[1].grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.6)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT / "fig5_triple_integration_pam50.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.2))
    specs = [
        ("Survival", "roc_auc", "A. Survival: triple-omics integration ROC AUC", "ROC AUC", (0.45, 0.85)),
        ("Survival", "c_index", "B. Survival: triple-omics integration C-index", "C-index", (0.45, 0.85)),
    ]
    x, labels, means, errs = scheme_bar(df, "Survival", "roc_auc", "", "")
    axes[0].bar(x, means, yerr=errs, capsize=2.5, color=TEAL, alpha=0.9, edgecolor="white", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_title(specs[0][2], loc="left", fontweight="bold", color=DARK)
    axes[0].set_ylim(*specs[0][4])
    axes[0].grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.6)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    x, labels, means, errs = scheme_bar(df, "Survival", "c_index", "", "")
    axes[1].bar(x, means, yerr=errs, capsize=2.5, color=ORANGE, alpha=0.9, edgecolor="white", linewidth=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("C-index")
    axes[1].set_title(specs[1][2], loc="left", fontweight="bold", color=DARK)
    axes[1].set_ylim(*specs[1][4])
    axes[1].grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.6)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT / "fig6_triple_integration_survival.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)


def fig_advanced_methods():
    files = {
        "Multi-task": ("advanced_method1_multitask_results.tsv",),
        "DeepSurv": ("advanced_method2_deepsurv_results.tsv",),
        "Transformer": ("advanced_method3_transformer_results.tsv",),
        "Low-rank bilinear": ("advanced_method4_lowrank_bilinear_results.tsv",),
        "GNN": ("advanced_method5_gnn_results.tsv",),
    }
    rows = []
    for method, (f,) in files.items():
        df = read_tsv(f)
        for task, metric in [("PAM50", "accuracy"), ("PAM50", "macro_f1"), ("Survival", "roc_auc"), ("Survival", "c_index")]:
            if "task" in df.columns:
                sub = df[(df["task"] == task) & (df["metric"] == metric)]
            else:
                sub = df[(df["metric"] == metric)]
            if not sub.empty:
                rows.append((method, task, metric, float(sub["mean"].iloc[0]), float(sub["std"].iloc[0])))
    d = pd.DataFrame(rows, columns=["method", "task", "metric", "mean", "std"])
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.8))
    panels = [
        ("PAM50", "accuracy", "A. PAM50: advanced integration Accuracy", "Accuracy", BLUE, (0.55, 1.0)),
        ("PAM50", "macro_f1", "B. PAM50: advanced integration Macro-F1", "Macro-F1", DARK, (0.55, 1.0)),
    ]
    methods = ["Multi-task", "Transformer", "Low-rank bilinear", "GNN"]
    for ax, (task, metric, title, ylabel, color, ylim) in zip(axes, panels):
        x = np.arange(len(methods))
        means, errs = [], []
        for m in methods:
            sub = d[(d["method"] == m) & (d["task"] == task) & (d["metric"] == metric)]
            means.append(float(sub["mean"].iloc[0]))
            errs.append(float(sub["std"].iloc[0]))
        ax.bar(x, means, yerr=errs, capsize=2.5, color=color, alpha=0.9, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold", color=DARK)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT / "fig7_advanced_methods_pam50.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.8))
    panels = [
        ("Survival", "roc_auc", "A. Survival: advanced integration ROC AUC", "ROC AUC", TEAL, (0.45, 0.80)),
        ("Survival", "c_index", "B. Survival: advanced integration C-index", "C-index", ORANGE, (0.45, 0.80)),
    ]
    methods = ["Multi-task", "DeepSurv", "Transformer", "Low-rank bilinear", "GNN"]
    for ax, (task, metric, title, ylabel, color, ylim) in zip(axes, panels):
        x = np.arange(len(methods))
        means, errs = [], []
        for m in methods:
            sub = d[(d["method"] == m) & (d["task"] == task) & (d["metric"] == metric)]
            means.append(float(sub["mean"].iloc[0]))
            errs.append(float(sub["std"].iloc[0]))
        ax.bar(x, means, yerr=errs, capsize=2.5, color=color, alpha=0.9, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold", color=DARK)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT / "fig8_advanced_methods_survival.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)


def fig_single_omics_combined():
    df = read_tsv("comprehensive_batch1_single_omics_results.tsv")
    models_pam = [
        "LogisticRegression",
        "RandomForest",
        "GradientBoosting",
        "SVC",
        "KNN",
        "MLP",
        "FullSizeCNN",
    ]
    models_sur = models_pam + ["CoxPH"]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.2))
    panels = [
        (axes[0, 0], df[df["task"] == "PAM50"], "accuracy", models_pam,
         "A. PAM50: Accuracy", "Accuracy", (0.45, 1.0)),
        (axes[0, 1], df[df["task"] == "PAM50"], "macro_f1", models_pam,
         "B. PAM50: Macro-F1", "Macro-F1", (0.40, 1.0)),
        (axes[1, 0], df[df["task"] == "Survival"], "roc_auc", models_sur,
         "C. Survival: ROC AUC", "ROC AUC", (0.45, 0.80)),
        (axes[1, 1], df[df["task"] == "Survival"], "c_index", models_sur,
         "D. Survival: C-index", "C-index", (0.45, 0.80)),
    ]
    for ax, sub, metric, models, title, ylabel, ylim in panels:
        grouped_bar(ax, sub, metric, ["mRNA", "CNV", "miRNA"], models, title, ylabel, ylim=ylim)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=7.5)
    # Build one compact legend containing all models.
    all_models = list(dict.fromkeys(models_pam + models_sur))
    palette = {
        "LogisticRegression": BLUE,
        "RandomForest": GREEN,
        "GradientBoosting": ORANGE,
        "SVC": PURPLE,
        "KNN": GRAY,
        "MLP": TEAL,
        "FullSizeCNN": RED,
        "CoxPH": DARK,
    }
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[m]) for m in all_models
    ]
    fig.legend(handles, all_models, loc="upper center", bbox_to_anchor=(0.5, 1.015),
               ncol=4, frameon=False, fontsize=7.5, handlelength=1.0, handleheight=0.7)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / "fig3_single_omics_combined.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)


def fig_triple_integration_combined():
    df = read_tsv("comprehensive_triple_integration_results.tsv")
    schemes = [
        "Concat_LogisticRegression",
        "Concat_RandomForest",
        "Concat_GradientBoosting",
        "Concat_SVC",
        "Concat_KNN",
        "Concat_MLP",
        "CNN_TripleConcat",
        "CNN_TripleGated",
        "CNN_LateAvg",
    ]
    labels = [s.replace("Concat_", "").replace("CNN_Triple", "CNN") for s in schemes]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.6))
    panels = [
        (axes[0, 0], "PAM50", "accuracy", "A. PAM50: Accuracy", "Accuracy", BLUE, (0.55, 1.0)),
        (axes[0, 1], "PAM50", "macro_f1", "B. PAM50: Macro-F1", "Macro-F1", DARK, (0.55, 1.0)),
        (axes[1, 0], "Survival", "roc_auc", "C. Survival: ROC AUC", "ROC AUC", TEAL, (0.45, 0.85)),
        (axes[1, 1], "Survival", "c_index", "D. Survival: C-index", "C-index", ORANGE, (0.45, 0.85)),
    ]
    for ax, task, metric, title, ylabel, color, ylim in panels:
        d = df[(df["task"] == task) & (df["metric"] == metric)]
        means, errs = [], []
        for s in schemes:
            sub = d[d["scheme"] == s]
            means.append(float(sub["mean"].iloc[0]))
            errs.append(float(sub["std"].iloc[0]))
        x = np.arange(len(schemes))
        ax.bar(x, means, yerr=errs, capsize=2.2, color=color, alpha=0.9,
               edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.0)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, loc="left", fontweight="bold", color=DARK, fontsize=9)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT / "fig4_triple_integration_combined.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)


def fig_advanced_methods_combined():
    files = {
        "Multi-task": ("advanced_method1_multitask_results.tsv",),
        "DeepSurv": ("advanced_method2_deepsurv_results.tsv",),
        "Transformer": ("advanced_method3_transformer_results.tsv",),
        "Low-rank bilinear": ("advanced_method4_lowrank_bilinear_results.tsv",),
        "GNN": ("advanced_method5_gnn_results.tsv",),
    }
    rows = []
    for method, (f,) in files.items():
        df = read_tsv(f)
        for task, metric in [("PAM50", "accuracy"), ("PAM50", "macro_f1"),
                             ("Survival", "roc_auc"), ("Survival", "c_index")]:
            if "task" in df.columns:
                sub = df[(df["task"] == task) & (df["metric"] == metric)]
            else:
                sub = df[(df["metric"] == metric)]
            if not sub.empty:
                rows.append((method, task, metric, float(sub["mean"].iloc[0]),
                             float(sub["std"].iloc[0])))
    d = pd.DataFrame(rows, columns=["method", "task", "metric", "mean", "std"])
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    panels = [
        (axes[0, 0], "PAM50", "accuracy", ["Multi-task", "Transformer", "Low-rank bilinear", "GNN"],
         "A. PAM50: Accuracy", "Accuracy", BLUE, (0.55, 1.0)),
        (axes[0, 1], "PAM50", "macro_f1", ["Multi-task", "Transformer", "Low-rank bilinear", "GNN"],
         "B. PAM50: Macro-F1", "Macro-F1", DARK, (0.55, 1.0)),
        (axes[1, 0], "Survival", "roc_auc", ["Multi-task", "DeepSurv", "Transformer", "Low-rank bilinear", "GNN"],
         "C. Survival: ROC AUC", "ROC AUC", TEAL, (0.45, 0.80)),
        (axes[1, 1], "Survival", "c_index", ["Multi-task", "DeepSurv", "Transformer", "Low-rank bilinear", "GNN"],
         "D. Survival: C-index", "C-index", ORANGE, (0.45, 0.80)),
    ]
    for ax, task, metric, methods, title, ylabel, color, ylim in panels:
        x = np.arange(len(methods))
        means, errs = [], []
        for m in methods:
            sub = d[(d["method"] == m) & (d["task"] == task) & (d["metric"] == metric)]
            means.append(float(sub["mean"].iloc[0]))
            errs.append(float(sub["std"].iloc[0]))
        ax.bar(x, means, yerr=errs, capsize=2.5, color=color, alpha=0.9,
               edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, loc="left", fontweight="bold", color=DARK, fontsize=9)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT / "fig5_advanced_methods_combined.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)


def main():
    fig_single_omics_combined()
    fig_triple_integration_combined()
    fig_advanced_methods_combined()


if __name__ == "__main__":
    main()
