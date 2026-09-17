#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a non-linear grid-style conceptual technical-route SVG."""

from pathlib import Path


OUT = Path(r"E:\组学数据图像化\data\paper_figures\fig1_concept_route_v5.svg")


def icon(key: str, color: str) -> str:
    s = f'<g transform="translate(0,0)" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">'
    if key == "db":
        s += (
            '<rect x="-30" y="-30" width="60" height="16" rx="6" fill="#E8F1F8"/>'
            '<path d="M-30 -22 h60"/>'
            '<rect x="-30" y="-8" width="60" height="16" rx="6" fill="#BBD5EA"/>'
            '<path d="M-30 0 h60"/>'
            '<rect x="-30" y="14" width="60" height="16" rx="6" fill="#D9EAF5"/>'
        )
    elif key == "wave":
        s += '<path d="M-32 0 Q-22 -18 -12 0 T8 0 T28 0" stroke-width="4"/>'
    elif key == "venn":
        s += '<circle cx="-9" cy="0" r="17"/><circle cx="9" cy="0" r="17"/>'
    elif key == "pam50":
        s += (
            '<rect x="-22" y="-22" width="20" height="20" rx="3" fill="#6AA4C9"/>'
            '<rect x="2" y="-22" width="20" height="20" rx="3" fill="#F2A45A"/>'
            '<rect x="-22" y="2" width="20" height="20" rx="3" fill="#D86C6C"/>'
            '<rect x="2" y="2" width="20" height="20" rx="3" fill="#81BBC8"/>'
        )
    elif key == "pulse":
        s += '<path d="M-32 0 C-22 -20 -8 -20 0 0 C6 14 16 14 22 0 C26 -8 30 -8 32 0"/>'
    elif key == "clock":
        s += '<circle cx="0" cy="0" r="24"/><path d="M0 -12 v12 l9 6" stroke-width="3"/>'
    elif key == "variance":
        s += '<path d="M-28 22 v-44 M-9 22 v-28 M10 22 v-36 M28 22 v-18"/>'
    elif key == "fl1":
        s += '<path d="M-30 22 h15 M-15 22 h15 M0 22 h15 M15 22 h15 M-30 10 h15 M0 10 h15" stroke-width="3"/>'
    elif key == "jsd":
        s += '<path d="M-26 24 V-16 h12 v40 M-8 24 V4 h12 v20 M10 24 V-4 h12 v28" stroke-width="3"/>'
    elif key == "sort":
        s += '<path d="M-24 -20 v40 M-32 12 l8 8 8 -8 M24 -20 v40 M16 12 l8 8 8 -8"/>'
    elif key == "spiral":
        s += '<path d="M0 0 C6 -2 12 0 10 6 C8 12 0 14 -6 10 C-12 6 -12 -2 -6 -8 C0 -14 12 -12 14 -4" stroke-width="3"/>'
    elif key == "maps":
        s += '<rect x="-28" y="-24" width="56" height="15" rx="4" fill="#D9EAF5"/><rect x="-28" y="-4" width="56" height="15" rx="4" fill="#BBD5EA"/><rect x="-28" y="16" width="56" height="15" rx="4" fill="#81BBC8"/>'
    elif key == "ml":
        s += '<path d="M0 -24 h0 v12 M0 -12 l-20 28 M0 -12 l20 28 M-20 16 l-10 16 M-20 16 l10 16 M20 16 l-10 16 M20 16 l10 16" stroke-width="3"/>'
    elif key == "mlp":
        s += '<circle cx="-22" cy="-12" r="7" fill="#F9D8AE"/><circle cx="-22" cy="12" r="7" fill="#F9D8AE"/><circle cx="20" cy="-12" r="7" fill="#F9D8AE"/><circle cx="20" cy="12" r="7" fill="#F9D8AE"/><circle cx="28" cy="0" r="7" fill="#F9D8AE"/><path d="M-15 -12 L13 -12 M-15 12 L13 12 M-22 -5 L-22 5 M20 -5 L20 5 M27 -12 L-15 5 M27 12 L-15 -5"/>'
    elif key == "cnn":
        s += '<rect x="-30" y="-30" width="60" height="60" rx="6"/><rect x="-13" y="-13" width="26" height="26" fill="#F2A45A"/>'
    elif key == "concat":
        s += '<path d="M-30 -20 L0 -8 M-30 0 L0 0 M-30 20 L0 8"/><circle cx="12" cy="0" r="15"/><path d="M27 0 L32 0"/>'
    elif key == "stack":
        s += '<rect x="-28" y="-24" width="56" height="14" rx="4" fill="#FAE8E8"/><rect x="-28" y="-5" width="56" height="14" rx="4" fill="#E39A9A"/><rect x="-28" y="14" width="56" height="14" rx="4" fill="#C86A6A"/>'
    elif key == "gauge":
        s += '<path d="M-28 20 A28 28 0 0 1 28 20"/><path d="M0 20 L-18 -6" stroke-width="3"/>'
    else:
        s += '<circle cx="0" cy="0" r="22"/>'
    s += "</g>"
    return s


def tile(x, y, key, label, color, light):
    w, h = 190, 160
    return (
        f'<g transform="translate({x},{y})">'
        f'<rect x="0" y="0" width="{w}" height="{h}" rx="14" fill="{light}" stroke="#C9D9E8" stroke-width="1.5"/>'
        f'<g transform="translate({w//2},68)">{icon(key,color)}</g>'
        f'<text x="{w//2}" y="146" text-anchor="middle" font-size="14" font-weight="700" fill="#24374E">{label}</text>'
        "</g>"
    )


def card(x, y, no, title, color, tiles):
    w, h = 650, 250
    parts = [
        f'<g transform="translate({x},{y})">',
        f'<rect x="0" y="0" width="{w}" height="{h}" rx="20" fill="#FFFFFF" stroke="#C9D9E8" stroke-width="1.5" filter="url(#shadow)"/>',
        f'<rect x="0" y="0" width="{w}" height="10" rx="5" fill="{color}"/>',
        f'<circle cx="34" cy="32" r="20" fill="#FFFFFF" stroke="{color}" stroke-width="2"/>',
        f'<text x="34" y="38" text-anchor="middle" font-size="16" font-weight="700" fill="{color}">{no}</text>',
        f'<text x="64" y="38" font-size="21" font-weight="700" fill="#1E3A5F">{title}</text>',
    ]
    tile_x = x + 22
    for i, (key, label, light) in enumerate(tiles):
        parts.append(tile(tile_x + i * 205, y + 65, key, label, color, light))
    parts.append("</g>")
    return "\n".join(parts)


def arrow(x0, y0, x1, y1):
    return f'<path d="M{x0} {y0} L{x1} {y1}" stroke="#7D9CB7" stroke-width="3" fill="none" marker-end="url(#arrow)"/>'


def main():
    cards_data = [
        (
            "1",
            "数据获取与样本对齐",
            "#2E6FA6",
            [("db", "TCGA-BRCA", "#E8F1F8"), ("wave", "mRNA/CNV/miRNA", "#E8F1F8"), ("venn", "统一交集样本", "#E8F1F8")],
        ),
        (
            "2",
            "临床标签定义",
            "#348FB3",
            [("pam50", "PAM50 四分类", "#E9F4F7"), ("pulse", "生存事件/删失", "#FDF1E2"), ("clock", "生存时间", "#E9F4F7")],
        ),
        (
            "3",
            "无泄漏特征筛选",
            "#60A5C7",
            [("variance", "低方差过滤", "#EAF4F8"), ("fl1", "F 值 / L1", "#EAF4F8"), ("jsd", "JSD 及组合", "#EAF4F8")],
        ),
        (
            "4",
            "JSD 组学图像化",
            "#81BBC8",
            [("sort", "重要性降序", "#EAF4F5"), ("spiral", "中心螺旋填充", "#EAF4F5"), ("maps", "三通道特征图", "#EAF4F5")],
        ),
        (
            "5",
            "单组学建模",
            "#F2A45A",
            [("ml", "传统 ML", "#FDF1E2"), ("mlp", "MLP", "#FDF1E2"), ("cnn", "FullSizeCNN", "#FDF1E2")],
        ),
        (
            "6",
            "多组学融合与评估",
            "#D86C6C",
            [("concat", "拼接/注意力", "#FAE8E8"), ("stack", "Stacking", "#FAE8E8"), ("gauge", "AUC/C-index", "#FAE8E8")],
        ),
    ]

    # Grid positions: left column top-down 1,2,3; right column bottom-up 4,5,6.
    left_x, right_x = 60, 790
    y1, y2, y3 = 150, 420, 690
    positions = [
        (left_x, y1),
        (left_x, y2),
        (left_x, y3),
        (right_x, y3),
        (right_x, y2),
        (right_x, y1),
    ]

    svg = []
    svg.append('<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1060" viewBox="0 0 1500 1060" font-family="Microsoft YaHei, Arial, sans-serif">')
    svg.append("<defs>")
    svg.append('<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="6" stdDeviation="9" flood-color="#1E3A5F" flood-opacity="0.12"/></filter>')
    svg.append('<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#F8FBFD"/><stop offset="1" stop-color="#EDF3F8"/></linearGradient>')
    svg.append('<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#7D9CB7"/></marker>')
    svg.append("</defs>")
    svg.append('<rect width="1500" height="1060" fill="url(#bg)"/>')
    svg.append('<text x="750" y="52" text-anchor="middle" font-size="30" font-weight="700" fill="#173A5E">JSD-Guided Omics Imaging and Multi-Omics Fusion</text>')
    svg.append('<text x="750" y="86" text-anchor="middle" font-size="17" fill="#5E6E83">基于 JSD 引导组学图像化与多组学融合的乳腺癌分子分型及生存预测 · 方法概念图</text>')

    for i, (no, title, color, tiles) in enumerate(cards_data):
        x, y = positions[i]
        svg.append(card(x, y, no, title, color, tiles))

    # U-shaped flow connectors.
    svg.append(arrow(385, 400, 385, 420))
    svg.append(arrow(385, 670, 385, 690))
    svg.append(arrow(710, 815, 790, 815))
    svg.append(arrow(1115, 690, 1115, 670))
    svg.append(arrow(1115, 420, 1115, 400))

    svg.append('<rect x="540" y="990" width="420" height="44" rx="22" fill="#FBF0F0" stroke="#D9A3A3"/>')
    svg.append('<text x="750" y="1019" text-anchor="middle" font-size="17" font-weight="700" fill="#A23E3E">下游可解释性：SHAP · Grad-CAM</text>')
    svg.append('<text x="750" y="1042" text-anchor="middle" font-size="13" fill="#6E7A8A">图 1. 研究方法概念图。特征筛选、标准化和模型拟合均在 5 折交叉验证训练折内完成。</text>')
    svg.append("</svg>")

    OUT.write_text("\n".join(svg), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
