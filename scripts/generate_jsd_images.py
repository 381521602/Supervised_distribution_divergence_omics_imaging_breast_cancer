#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build grayscale omics images ordered by JSD."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_multistream_cnn import jsd_scores, spiral_order  # noqa: E402


PAM_DIR = ROOT / "data" / "final_datasets" / "PAM50"
SUR_DIR = ROOT / "data" / "final_datasets" / "Survival"
IMG_ROOT = ROOT / "data" / "images"


def make_images(X, y, size):
    scores = jsd_scores(X, y)
    order = np.argsort(scores)[::-1]
    positions = spiral_order(size)
    n = X.shape[0]
    imgs = np.zeros((n, 1, size, size), dtype=np.float32)
    for i in range(n):
        for pos, feat_idx in zip(positions, order):
            if feat_idx < X.shape[1]:
                imgs[i, 0, pos[0], pos[1]] = X[i, feat_idx]
    return imgs, order, scores


def process(task_dir, task_name, config, label_cols):
    for omics, (size, label_col) in config.items():
        df = pd.read_csv(task_dir / f"{omics}_{task_name}_final.tsv", sep="\t")
        X = df.drop(columns=label_cols).to_numpy(dtype=float)
        y = df[label_col].values if label_col != "survival" else df["os_event"].astype(int).values
        imgs, order, scores = make_images(X, y, size)
        out_dir = IMG_ROOT / task_name / omics
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "images.npy", imgs)
        np.save(out_dir / "order.npy", order)
        pd.DataFrame({"feature": df.drop(columns=label_cols).columns[order], "jsd": scores[order]}).to_csv(out_dir / "order.tsv", sep="\t", index=False)
        print(f"{task_name}/{omics}: {imgs.shape}")


def main():
    process(PAM_DIR, "PAM50", {
        "mRNA": (20, "pam50"),
        "CNV": (8, "pam50"),
        "miRNA": (25, "pam50"),
    }, ["pam50"])
    process(SUR_DIR, "Survival", {
        "mRNA": (15, "os_event"),
        "CNV": (13, "os_event"),
        "miRNA": (25, "os_event"),
    }, ["case_id", "os_event", "os_time_days"])


if __name__ == "__main__":
    main()
