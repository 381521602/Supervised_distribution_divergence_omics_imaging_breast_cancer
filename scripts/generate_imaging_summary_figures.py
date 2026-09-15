#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate summary figures for the imaging prediction report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "images/report_figures"
OUT.mkdir(parents=True, exist_ok=True)


def read(path):
    return pd.read_csv(DATA / path, sep="\t")


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def fig_single_omics():
    df = read("wgan_gp_augmentation_all_omics_results.tsv")
    base = df[df["variant"] == "baseline"]
    pam_acc = []
    pam_f1 = []
    sur_auc = []
    sur_ci = []
    omics = ["mRNA", "CNV", "miRNA"]
    for o in omics:
        pam_acc.append(base[(base.omics == o) & (base.task == "PAM50") & (base.metric == "accuracy")]["mean"].iloc[0])
        pam_f1.append(base[(base.omics == o) & (base.task == "PAM50") & (base.metric == "macro_f1")]["mean"].iloc[0])
        sur_auc.append(base[(base.omics == o) & (base.task == "Survival") & (base.metric == "roc_auc")]["mean"].iloc[0])
        sur_ci.append(base[(base.omics == o) & (base.task == "Survival") & (base.metric == "c_index")]["mean"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(omics))
    w = 0.35
    axes[0].bar(x - w / 2, pam_acc, w, label="Accuracy", color="#2E74B5")
    axes[0].bar(x + w / 2, pam_f1, w, label="Macro-F1", color="#7FA8D9")
    axes[0].set_title("PAM50: single-omics FullSizeCNN")
    axes[0].set_xticks(x, omics)
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    axes[1].bar(x - w / 2, sur_auc, w, label="ROC AUC", color="#1F4D78")
    axes[1].bar(x + w / 2, sur_ci, w, label="C-index", color="#8FAADC")
    axes[1].set_title("Survival: single-omics FullSizeCNN")
    axes[1].set_xticks(x, omics)
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)
    style_axis(axes[1])

    fig.suptitle("Single-omics FullSizeCNN baseline", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_single_omics_baseline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_classic_structures():
    df = read("nsre_classic_structures_results.tsv")
    models = ["FullSizeCNN", "GroupNorm_Mish_FullSizeCNN", "FCN_FullSize", "ResNet_FullSize", "MultiBranch_SE_FullSize"]
    pam_acc = []
    pam_f1 = []
    sur_auc = []
    sur_ci = []
    for m in models:
        pam_acc.append(df[(df.variant == m) & (df.task == "PAM50") & (df.metric == "accuracy")]["mean"].iloc[0])
        pam_f1.append(df[(df.variant == m) & (df.task == "PAM50") & (df.metric == "macro_f1")]["mean"].iloc[0])
        sur_auc.append(df[(df.variant == m) & (df.task == "Survival") & (df.metric == "roc_auc")]["mean"].iloc[0])
        sur_ci.append(df[(df.variant == m) & (df.task == "Survival") & (df.metric == "c_index")]["mean"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(models))
    w = 0.35
    axes[0].bar(x - w / 2, pam_acc, w, label="Accuracy", color="#2E74B5")
    axes[0].bar(x + w / 2, pam_f1, w, label="Macro-F1", color="#7FA8D9")
    axes[0].set_title("PAM50 classification")
    axes[0].set_xticks(x, ["Base", "GN+Mish", "FCN", "ResNet", "SE"], rotation=15)
    axes[0].set_ylim(0.8, 1)
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    axes[1].bar(x - w / 2, sur_auc, w, label="ROC AUC", color="#1F4D78")
    axes[1].bar(x + w / 2, sur_ci, w, label="C-index", color="#8FAADC")
    axes[1].set_title("Survival prediction")
    axes[1].set_xticks(x, ["Base", "GN+Mish", "FCN", "ResNet", "SE"], rotation=15)
    axes[1].set_ylim(0.55, 0.8)
    axes[1].legend(frameon=False)
    style_axis(axes[1])

    fig.suptitle("Classic full-size convolutional structures", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_classic_structures.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_integration():
    df = read("fullsize_multimodal_integration_results.tsv")
    schemes = ["single_mRNA", "triple_concat", "triple_gated", "mrna_mirna_concat", "late_fusion_avg"]
    labels = ["mRNA only", "Triple concat", "Triple gated", "mRNA+miRNA", "Late fusion"]
    pam_acc = [df[(df.scheme == s) & (df.task == "PAM50") & (df.metric == "accuracy")]["mean"].iloc[0] for s in schemes]
    pam_f1 = [df[(df.scheme == s) & (df.task == "PAM50") & (df.metric == "macro_f1")]["mean"].iloc[0] for s in schemes]
    sur_auc = [df[(df.scheme == s) & (df.task == "Survival") & (df.metric == "roc_auc")]["mean"].iloc[0] for s in schemes]
    sur_ci = [df[(df.scheme == s) & (df.task == "Survival") & (df.metric == "c_index")]["mean"].iloc[0] for s in schemes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(schemes))
    w = 0.35
    axes[0].bar(x - w / 2, pam_acc, w, label="Accuracy", color="#2E74B5")
    axes[0].bar(x + w / 2, pam_f1, w, label="Macro-F1", color="#7FA8D9")
    axes[0].set_title("PAM50 common-sample integration")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylim(0.8, 1)
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    axes[1].bar(x - w / 2, sur_auc, w, label="ROC AUC", color="#1F4D78")
    axes[1].bar(x + w / 2, sur_ci, w, label="C-index", color="#8FAADC")
    axes[1].set_title("Survival common-sample integration")
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylim(0.6, 0.85)
    axes[1].legend(frameon=False)
    style_axis(axes[1])

    fig.suptitle("FullSizeCNN multi-omics integration", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_integration.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_augmentation():
    df = read("wgan_gp_augmentation_all_omics_results.tsv")
    variants = ["baseline", "aug_1x", "aug_5x", "aug_10x"]
    labels = ["None", "1x", "5x", "10x"]
    omics = ["mRNA", "CNV", "miRNA"]
    colors = {"mRNA": "#2E74B5", "CNV": "#D68A2A", "miRNA": "#2F8F5B"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for o in omics:
        vals = [df[(df.omics == o) & (df.task == "PAM50") & (df.variant == v) & (df.metric == "accuracy")]["mean"].iloc[0] for v in variants]
        axes[0].plot(labels, vals, marker="o", label=o, color=colors[o])
    axes[0].set_title("PAM50: WGAN-GP augmentation effect")
    axes[0].set_ylim(0.55, 1)
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    for o in omics:
        vals = [df[(df.omics == o) & (df.task == "Survival") & (df.variant == v) & (df.metric == "c_index")]["mean"].iloc[0] for v in variants]
        axes[1].plot(labels, vals, marker="o", label=o, color=colors[o])
    axes[1].set_title("Survival: WGAN-GP augmentation effect")
    axes[1].set_ylim(0.5, 0.8)
    axes[1].legend(frameon=False)
    style_axis(axes[1])

    fig.suptitle("WGAN-GP augmentation by omics and augmentation level", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_augmentation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_single_omics()
    fig_classic_structures()
    fig_integration()
    fig_augmentation()
    print("figures saved ->", OUT)


if __name__ == "__main__":
    main()
