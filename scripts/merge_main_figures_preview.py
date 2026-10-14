#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compose previews of the consolidated main-text figures (no text replacement)."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"E:\组学数据图像化")
FIG = ROOT / "data" / "paper_figures"
OUT = FIG / "merged_preview"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_WIDTH = 4000
GAP = 70
LABEL_H = 110


def font(size):
    for p in (r"C:\Windows\Fonts\timesbd.ttf", r"C:\Windows\Fonts\times.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def panel(img_path, letter, width):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    nh = round(width * h / w)
    img = img.resize((width, nh), Image.LANCZOS)
    block = Image.new("RGB", (width, LABEL_H + nh), "white")
    d = ImageDraw.Draw(block)
    d.text((24, 14), letter, fill="black", font=font(96))
    block.paste(img, (0, LABEL_H))
    return block


def panel_height(img_path, letter, height):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    nw = round(height * w / h)
    img = img.resize((nw, height), Image.LANCZOS)
    block = Image.new("RGB", (nw, LABEL_H + height), "white")
    d = ImageDraw.Draw(block)
    d.text((24, 14), letter, fill="black", font=font(96))
    block.paste(img, (0, LABEL_H))
    return block


def label_block(letter, width):
    block = Image.new("RGB", (width, LABEL_H), "white")
    d = ImageDraw.Draw(block)
    d.text((24, 14), letter, fill="black", font=font(96))
    return block


def panel_square(img_path, letter, side):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((side, side), Image.LANCZOS)
    block = Image.new("RGB", (side, LABEL_H + side), "white")
    d = ImageDraw.Draw(block)
    d.text((24, 14), letter, fill="black", font=font(96))
    block.paste(img, (0, LABEL_H))
    return block


def panel_custom(img_path, letter, width, height, label_h=LABEL_H, label_size=96):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    block = Image.new("RGB", (width, label_h + height), "white")
    d = ImageDraw.Draw(block)
    d.text((24, max(6, label_h - 82)), letter, fill="black", font=font(label_size))
    block.paste(img, (0, label_h))
    return block


def stack(blocks, out_name):
    width = max(b.width for b in blocks)
    total_h = sum(b.height for b in blocks) + GAP * (len(blocks) - 1)
    canvas = Image.new("RGB", (width, total_h), "white")
    y = 0
    for b in blocks:
        canvas.paste(b, ((width - b.width) // 2, y))
        y += b.height + GAP
    canvas.save(OUT / out_name, "PNG")
    print(out_name, canvas.size)


# New Figure 2: omics image representation (old Fig2 + old Fig3).
stack(
    [
        panel(FIG / "fig2_2d_detail.png", "A", TARGET_WIDTH),
        panel(FIG / "fig3_example_v12_condensed.png", "B", TARGET_WIDTH),
    ],
    "fig2_method_image_representation.png",
)

# New Figure 3: prediction performance (old Fig4 + old Fig5 + old Fig6).
top = panel(FIG / "fig3_single_omics_v4_merged.png", "A", TARGET_WIDTH)

# Two transposed heatmaps side by side at equal height under the single-omics panel.
left = panel_height(FIG / "fig5_triple_integration_heatmap_T.png", "B", 860)
right = panel_height(FIG / "fig6_advanced_methods_heatmap_T.png", "C", 860)
bottom_w = left.width + GAP + right.width
bottom_h = max(left.height, right.height)
bottom = Image.new("RGB", (bottom_w, bottom_h), "white")
bottom.paste(left, (0, 0))
bottom.paste(right, (left.width + GAP, 0))

fig3 = Image.new(
    "RGB",
    (TARGET_WIDTH, top.height + GAP + bottom.height),
    "white",
)
fig3.paste(top, (0, 0))
fig3.paste(bottom, ((TARGET_WIDTH - bottom_w) // 2, top.height + GAP))
fig3.save(OUT / "fig3_prediction_performance.png", "PNG")
print("fig3_prediction_performance.png", fig3.size)

# New Figure 4: A/B vertical columns on the left; B workflow + shorter C on the right.
WFLOW = 1400
_wf = Image.open(FIG / "masking_workflow_strip_v2.png").convert("RGB")
_wfw, _wfh = _wf.size
wf = _wf.resize((WFLOW, round(WFLOW * _wfh / _wfw)), Image.LANCZOS)
C_W = 1320
fig4_c = panel_custom(FIG / "fig4_masking_heatmap_combined_square.png", "C", C_W, 1260, label_h=70, label_size=72)

overlap = 240
right_h = LABEL_H + (wf.height - overlap) + fig4_c.height
right_col = Image.new("RGB", (WFLOW, right_h), "white")
right_col.paste(wf, (0, LABEL_H))
right_col.paste(fig4_c, ((WFLOW - C_W) // 2, LABEL_H + wf.height - overlap))

fig4_a = panel_height(FIG / "fig7_attribution_v6_vertical.png", "A", 2800)
fig4_b = panel_height(ROOT / "data" / "interpretability" / "figures" / "fig_scheme2_masking_flow_en_v5_vertical.png", "B", 2800)

fig4_h = max(fig4_a.height, fig4_b.height, right_h)
content_w = fig4_a.width + GAP + fig4_b.width + GAP + WFLOW
fig4 = Image.new("RGB", (TARGET_WIDTH, fig4_h), "white")
x0 = (TARGET_WIDTH - content_w) // 2
fig4.paste(fig4_a, (x0, 0))
fig4.paste(fig4_b, (x0 + fig4_a.width + GAP, 0))
fig4.paste(right_col, (x0 + fig4_a.width + GAP + fig4_b.width + GAP, 0))
fig4.save(OUT / "fig4_interpretability_masking.png", "PNG")
print("fig4_interpretability_masking.png", fig4.size)
