#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compose compact multi-panel overview/example figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)


def compose_two(path_a, path_b, out_name, figsize, titles=("A", "B"), hspace=0.02):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, path, title in zip(axes, [path_a, path_b], titles):
        img = imread(str(path))
        ax.imshow(img, interpolation="nearest", aspect="auto")
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=2)
    fig.subplots_adjust(left=0.002, right=0.998, top=0.90, bottom=0.005, wspace=hspace)
    out = OUT / out_name
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(out)


def compose_vertical(path_a, path_b, out_name, figsize, titles=("A", "B"), hspace=0.02):
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for ax, path, title in zip(axes, [path_a, path_b], titles):
        img = imread(str(path))
        ax.imshow(img, interpolation="nearest", aspect="auto")
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=2)
    fig.subplots_adjust(left=0.002, right=0.998, top=0.965, bottom=0.005, hspace=hspace)
    out = OUT / out_name
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(out)


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def split_three(path) -> list[Image.Image]:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    third = w // 3
    panels = []
    for i in range(3):
        crop = img.crop((i * third, 0, (i + 1) * third, h))
        panels.append(center_crop_square(crop))
    return panels


def compose_square_triples(path_a, path_b, out_name, figsize=(8.6, 6.2)):
    panels_a = split_three(path_a)
    panels_b = split_three(path_b)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    labels = [
        ("A1", "A2", "A3"),
        ("B1", "B2", "B3"),
    ]
    for r, panels in enumerate([panels_a, panels_b]):
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, interpolation="nearest", aspect="equal")
            ax.set_axis_off()
            ax.set_title(labels[r][c], loc="left", fontsize=10, fontweight="bold", pad=2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.005, wspace=0.03, hspace=0.03)
    out = OUT / out_name
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(out)


def main():
    compose_square_triples(
        ROOT / "data/images/examples/PAM50_mRNA_triple.png",
        ROOT / "data/images/examples/Survival_mRNA_triple.png",
        "fig2_example_triples.png",
        figsize=(8.6, 6.2),
    )


if __name__ == "__main__":
    main()
