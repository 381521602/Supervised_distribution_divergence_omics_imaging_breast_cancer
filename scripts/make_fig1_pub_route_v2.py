#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a high-density publication-style method overview figure.

The figure combines a drawn six-stage technical pipeline with representative
project figures (omics image examples, NSRE visualization, single-omics results,
triple-omics integration results, and advanced-method results).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(r"E:\组学数据图像化")
FIG_DIR = ROOT / "data" / "paper_figures"
OUT = FIG_DIR / "fig1_method_overview_pub_v2.png"

W, H = 3840, 2460
MARGIN = 70
BG = (255, 255, 255)
INK = (18, 38, 63)
MUTED = (75, 90, 110)
LINE = (205, 216, 228)
PANEL_BG = (247, 250, 253)

FONT_REGULAR = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy,
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill=INK,
    anchor="la",
):
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def draw_arrow(draw: ImageDraw.ImageDraw, x0: int, y: int, x1: int, color=INK, width: int = 6):
    draw.line((x0, y, x1, y), fill=color, width=width)
    head = 24
    draw.polygon(
        [(x1, y), (x1 - head, y - head // 2), (x1 - head, y + head // 2)],
        fill=color,
    )


def fit_preserve(
    img: Image.Image,
    target_w: int,
    target_h: int,
    bg=(255, 255, 255),
) -> Image.Image:
    """Resize an image to fit inside target box, preserving aspect ratio."""
    img = ImageOps.exif_transpose(img).convert("RGBA")
    canvas = Image.new("RGBA", (target_w, target_h), bg + (255,))
    ratio = min(target_w / img.width, target_h / img.height)
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas.alpha_composite(img, (x, y))
    return canvas.convert("RGB")


def load(name: str) -> Image.Image:
    return Image.open(FIG_DIR / name)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy,
    fill,
    outline,
    radius: int = 26,
    width: int = 4,
):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_pipeline(draw: ImageDraw.ImageDraw):
    modules = [
        {
            "no": "01",
            "title": "数据获取与样本对齐",
            "color": (44, 110, 174),
            "light": (231, 241, 249),
            "items": [
                "TCGA-BRCA",
                "mRNA / CNV / miRNA",
                "统一交集样本",
            ],
        },
        {
            "no": "02",
            "title": "标签定义",
            "color": (52, 143, 179),
            "light": (233, 245, 248),
            "items": [
                "PAM50 四分类",
                "总体生存 / 事件",
                "OS 时间与删失",
            ],
        },
        {
            "no": "03",
            "title": "无泄漏特征筛选",
            "color": (96, 165, 199),
            "light": (238, 247, 250),
            "items": [
                "低方差 / F / L1",
                "NSRE 对称相对熵",
                "组合与数量扩增",
            ],
        },
        {
            "no": "04",
            "title": "NSRE 组学图像化",
            "color": (129, 187, 216),
            "light": (240, 248, 251),
            "items": [
                "重要性降序排序",
                "中心向外螺旋填充",
                "灰度 / 功能 / NSRE 图",
            ],
        },
        {
            "no": "05",
            "title": "单组学建模",
            "color": (242, 164, 90),
            "light": (253, 245, 235),
            "items": [
                "LR / RF / GBM / SVC / KNN",
                "MLP",
                "FullSizeCNN",
            ],
        },
        {
            "no": "06",
            "title": "多组学融合与验证",
            "color": (216, 108, 108),
            "light": (251, 239, 239),
            "items": [
                "拼接 / 注意力 / Stacking",
                "5 折交叉验证",
                "Acc / Macro-F1 / AUC / C-index",
            ],
        },
    ]

    y0, y1 = 250, 800
    gap = 28
    box_w = (W - 2 * MARGIN - gap * (len(modules) - 1)) // len(modules)
    x = MARGIN
    y_center = (y0 + y1) // 2
    for i, mod in enumerate(modules):
        x1 = x + box_w
        rounded_box(
            draw,
            (x, y0, x1, y1),
            fill=mod["light"],
            outline=mod["color"],
            radius=24,
            width=5,
        )
        # Header band
        draw.rounded_rectangle(
            (x, y0, x1, y0 + 104),
            radius=24,
            fill=mod["color"],
        )
        draw.rectangle((x, y0 + 55, x1, y0 + 104), fill=mod["color"])
        draw_text(
            draw,
            (x + 24, y0 + 34),
            mod["no"],
            font(30, bold=True),
            fill=(255, 255, 255),
        )
        draw_text(
            draw,
            (x1 - 24, y0 + 40),
            mod["title"],
            font(31, bold=True),
            fill=(255, 255, 255),
            anchor="ra",
        )
        text_y = y0 + 152
        for item in mod["items"]:
            draw_text(draw, (x + 24, text_y), "•  " + item, font(26), fill=INK)
            text_y += 54
        # Downstream labels
        if i < len(modules) - 1:
            draw_arrow(draw, x1 + 4, y_center, x1 + gap - 4, color=mod["color"], width=7)
        x = x1 + gap

    # Small interpretability extension on the right of module 06
    ext_x0 = MARGIN + 5 * (box_w + gap) + 4
    draw_text(
        draw,
        (W - MARGIN, y_center + 118),
        "SHAP / Grad-CAM",
        font(26, bold=True),
        fill=(180, 70, 70),
        anchor="ra",
    )
    draw_text(
        draw,
        (W - MARGIN, y_center + 160),
        "关键因子与通路挖掘",
        font(26, bold=True),
        fill=(180, 70, 70),
        anchor="ra",
    )


def panel_frame(
    draw: ImageDraw.ImageDraw,
    x0,
    y0,
    x1,
    y1,
    label: str,
    title: str,
    title_color,
):
    rounded_box(draw, (x0, y0, x1, y1), fill=PANEL_BG, outline=LINE, radius=22, width=4)
    draw_text(draw, (x0 + 24, y0 + 26), label, font(28, bold=True), fill=title_color)
    draw_text(draw, (x0 + 66, y0 + 32), title, font(26, bold=True), fill=INK)


def draw_panels(canvas: Image.Image, draw: ImageDraw.ImageDraw):
    top = 925
    bottom = H - 55
    gap = 28
    panel_w = (W - 2 * MARGIN - gap * 2) // 3
    x0 = MARGIN
    x1 = x0 + panel_w
    x2 = x1 + gap + panel_w
    x3 = x2 + gap + panel_w

    caption_h = 88
    img_y = top + caption_h
    img_h = bottom - img_y

    palette = [
        (44, 110, 174),
        (52, 143, 179),
        (96, 165, 199),
        (129, 187, 216),
        (242, 164, 90),
        (216, 108, 108),
    ]

    # Panel A
    panel_frame(
        draw,
        x0,
        top,
        x1,
        bottom,
        "A",
        "三组学图像化示例",
        palette[0],
    )
    im = fit_preserve(load("fig2_example_triples.png"), panel_w - 24, img_h - 20)
    draw.rectangle((x0 + 12, img_y + 10, x1 - 12, bottom - 10), outline=LINE, width=3)
    canvas.paste(im, (x0 + 12, img_y + 10))

    # Panel B
    panel_frame(
        draw,
        x1,
        top,
        x2,
        bottom,
        "B",
        "NSRE 图像化特征挖掘可视化",
        palette[1],
    )
    im = fit_preserve(load("fig6_nsre_annotated.png"), panel_w - 24, img_h - 20)
    draw.rectangle((x1 + 12, img_y + 10, x2 - 12, bottom - 10), outline=LINE, width=3)
    canvas.paste(im, (x1 + 12, img_y + 10))

    # Panel C
    panel_frame(
        draw,
        x2,
        top,
        x3,
        bottom,
        "C",
        "单组学 PAM50 预测结果",
        palette[2],
    )
    im = fit_preserve(load("fig3_single_omics_combined.png"), panel_w - 24, img_h - 20)
    draw.rectangle((x2 + 12, img_y + 10, x3 - 12, bottom - 10), outline=LINE, width=3)
    canvas.paste(im, (x2 + 12, img_y + 10))


def draw_panels_lower(canvas: Image.Image, draw: ImageDraw.ImageDraw):
    top = 925
    bottom = H - 55
    gap = 28
    panel_w = (W - 2 * MARGIN - gap * 2) // 3
    x0 = MARGIN
    x1 = x0 + panel_w
    x2 = x1 + gap + panel_w
    x3 = x2 + gap + panel_w
    caption_h = 88
    img_y = top + caption_h
    img_h = bottom - img_y

    # Panel D
    panel_frame(
        draw,
        x0,
        top,
        x1,
        bottom,
        "D",
        "三组学融合预测结果",
        (129, 187, 216),
    )
    im = fit_preserve(load("fig4_triple_integration_combined.png"), panel_w - 24, img_h - 20)
    draw.rectangle((x0 + 12, img_y + 10, x1 - 12, bottom - 10), outline=LINE, width=3)
    canvas.paste(im, (x0 + 12, img_y + 10))

    # Panel E
    panel_frame(
        draw,
        x1,
        top,
        x2,
        bottom,
        "E",
        "复杂 / 高级方法综合对比",
        (242, 164, 90),
    )
    im = fit_preserve(load("fig5_advanced_methods_combined.png"), panel_w - 24, img_h - 20)
    draw.rectangle((x1 + 12, img_y + 10, x2 - 12, bottom - 10), outline=LINE, width=3)
    canvas.paste(im, (x1 + 12, img_y + 10))

    # Panel F: two stacked advanced-result thumbnails.
    panel_frame(
        draw,
        x2,
        top,
        x3,
        bottom,
        "F",
        "PAM50 / 生存预测结果",
        (216, 108, 108),
    )
    half_h = (img_h - 32) // 2
    left = x2 + 12
    right = x3 - 12
    draw.rectangle((left, img_y + 10, right, img_y + 12 + half_h), outline=LINE, width=3)
    im = fit_preserve(load("fig7_advanced_methods_pam50.png"), right - left - 6, half_h - 6)
    canvas.paste(im, (left + 3, img_y + 13))
    draw_text(draw, (left + 14, img_y + 18), "PAM50", font(22, bold=True), fill=(216, 108, 108))

    y2 = img_y + 18 + half_h
    draw.rectangle((left, y2, right, bottom - 10), outline=LINE, width=3)
    im = fit_preserve(load("fig8_advanced_methods_survival.png"), right - left - 6, half_h - 6)
    canvas.paste(im, (left + 3, y2 + 3))
    draw_text(draw, (left + 14, y2 + 8), "Survival", font(22, bold=True), fill=(216, 108, 108))


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Title
    draw_text(
        draw,
        (MARGIN, 42),
        "NSRE-guided Omics Imaging and Multi-omics Fusion for Breast Cancer Molecular Subtyping and Survival Prediction",
        font(46, bold=True),
        fill=INK,
    )
    draw_text(
        draw,
        (MARGIN, 120),
        "基于 NSRE 引导组学图像化与多组学融合的乳腺癌分子分型及生存预测 · 方法与结果总览",
        font(32),
        fill=MUTED,
    )
    draw.line((MARGIN, 205, W - MARGIN, 205), fill=LINE, width=4)

    draw_pipeline(draw)
    draw_panels(img, draw)
    draw_panels_lower(img, draw)

    # Footer legend / note
    draw_text(
        draw,
        (MARGIN, H - 20),
        "图 1. 方法总览。A–F 为代表性中间结果；所有模型结果均为基于预选特征、5 折交叉验证的均值±标准差。",
        font(26),
        fill=MUTED,
    )

    img.save(OUT, dpi=(300, 300))
    print(OUT)


if __name__ == "__main__":
    main()
