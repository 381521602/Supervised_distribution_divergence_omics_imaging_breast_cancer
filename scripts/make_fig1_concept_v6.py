#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a clean 3x2 grid conceptual technical-route SVG with large sub-module rows."""

from pathlib import Path


OUT = Path(r"E:\组学数据图像化\data\paper_figures\fig1_concept_route_v6.svg")


def icon(key: str, color: str) -> str:
    s = f'<g transform="translate(0,0)" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">'
    if key == "db":
        s += (
            '<rect x="-30" y="-26" width="60" height="14" rx="6" fill="#E8F1F8"/>'
            '<path d="M-30 -19 h60"/>'
            '<rect x="-30" y="-6" width="60" height="14" rx="6" fill="#BBD5EA"/>'
            '<path d="M-30 1 h60"/>'
            '<rect x="-30" y="14" width="60" height="14" rx="6" fill="#D9EAF5"/>'
        )
    elif key == "wave":
        s += '<path d="M-30 0 Q-20 -16 -10 0 T10 0 T30 0" stroke-width="4"/>'
    elif key == "venn":
        s += '<circle cx="-8" cy="0" r="16"/><circle cx="8" cy="0" r="16"/>'
    elif key == "pam50":
        s += '<rect x="-20" y="-20" width="18" height="18" rx="3" fill="#6AA4C9"/><rect x="2" y="-20" width="18" height="18" rx="3" fill="#F2A45A"/><rect x="-20" y="2" width="18" height="18" rx="3" fill="#D86C6C"/><rect x="2" y="2" width="18" height="18" rx="3" fill="#81BBC8"/>'
    elif key == "pulse":
        s += '<path d="M-30 0 C-20 -18 -6 -18 0 0 C5 12 15 12 20 0 C24 -8 28 -8 30 0"/>'
    elif key == "clock":
        s += '<circle cx="0" cy="0" r="22"/><path d="M0 -11 v11 l8 6" stroke-width="3"/>'
    elif key == "variance":
        s += '<path d="M-26 20 v-40 M-9 20 v-26 M9 20 v-34 M26 20 v-16"/>'
    elif key == "fl1":
        s += '<path d="M-28 20 h13 M-15 20 h13 M-2 20 h13 M11 20 h13 M-28 8 h13 M-2 8 h13" stroke-width="3"/>'
    elif key == "jsd":
        s += '<path d="M-24 22 V-14 h11 v36 M-6 22 V4 h11 v18 M12 22 V-4 h11 v26" stroke-width="3"/>'
    elif key == "sort":
        s += '<path d="M-22 -18 v38 M-30 11 l8 9 8 -9 M22 -18 v38 M14 11 l8 9 8 -9"/>'
    elif key == "spiral":
        s += '<path d="M0 0 C5 -2 11 0 9 6 C7 11 0 13 -6 9 C-11 5 -11 -2 -6 -7 C0 -12 11 -11 13 -4" stroke-width="3"/>'
    elif key == "maps":
        s += '<rect x="-26" y="-22" width="52" height="14" rx="4" fill="#D9EAF5"/><rect x="-26" y="-3" width="52" height="14" rx="4" fill="#BBD5EA"/><rect x="-26" y="16" width="52" height="14" rx="4" fill="#81BBC8"/>'
    elif key == "ml":
        s += '<path d="M0 -22 v11 M0 -11 l-18 26 M0 -11 l18 26 M-18 15 l-9 15 M-18 15 l9 15 M18 15 l-9 15 M18 15 l9 15" stroke-width="3"/>'
    elif key == "mlp":
        s += '<circle cx="-20" cy="-10" r="7" fill="#F9D8AE"/><circle cx="-20" cy="10" r="7" fill="#F9D8AE"/><circle cx="18" cy="-10" r="7" fill="#F9D8AE"/><circle cx="18" cy="10" r="7" fill="#F9D8AE"/><circle cx="26" cy="0" r="7" fill="#F9D8AE"/><path d="M-13 -10 L11 -10 M-13 10 L11 10 M-20 -3 L-20 3 M18 -3 L18 3 M25 -10 L-13 3 M25 10 L-13 -3"/>'
    elif key == "cnn":
        s += '<rect x="-28" y="-28" width="56" height="56" rx="6"/><rect x="-12" y="-12" width="24" height="24" fill="#F2A45A"/>'
    elif key == "concat":
        s += '<path d="M-28 -18 L0 -8 M-28 0 L0 0 M-28 18 L0 8"/><circle cx="10" cy="0" r="14"/><path d="M24 0 L30 0"/>'
    elif key == "stack":
        s += '<rect x="-26" y="-22" width="52" height="13" rx="4" fill="#FAE8E8"/><rect x="-26" y="-4" width="52" height="13" rx="4" fill="#E39A9A"/><rect x="-26" y="14" width="52" height="13" rx="4" fill="#C86A6A"/>'
    elif key == "gauge":
        s += '<path d="M-26 18 A26 26 0 0 1 26 18"/><path d="M0 18 L-16 -5" stroke-width="3"/>'
    else:
        s += '<circle cx="0" cy="0" r="20"/>'
    s += "</g>"
    return s


def subrow(x, y, key, label, color, light):
    return (
        f'<g transform="translate({x},{y})">'
        f'<rect x="0" y="0" width="396" height="72" rx="12" fill="{light}" stroke="#C9D9E8" stroke-width="1.5"/>'
        f'<g transform="translate(46,36)">{icon(key,color)}</g>'
        f'<text x="86" y="43" font-size="15.5" font-weight="700" fill="#24374E">{label}</text>'
        "</g>"
    )


def card(x, y, no, title, color, rows):
    w, h = 440, 360
    parts = [
        f'<g transform="translate({x},{y})">',
        f'<rect x="0" y="0" width="{w}" height="{h}" rx="20" fill="#FFFFFF" stroke="#C9D9E8" stroke-width="1.5" filter="url(#shadow)"/>',
        f'<rect x="0" y="0" width="{w}" height="10" rx="5" fill="{color}"/>',
        f'<circle cx="34" cy="32" r="20" fill="#FFFFFF" stroke="{color}" stroke-width="2"/>',
        f'<text x="34" y="38" text-anchor="middle" font-size="16" font-weight="700" fill="{color}">{no}</text>',
        f'<text x="64" y="39" font-size="19" font-weight="700" fill="#1E3A5F">{title}</text>',
    ]
    for i, (key, label, light) in enumerate(rows):
        parts.append(subrow(x + 22, y + 62 + i * 82, key, label, color, light))
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
            [("db", "TCGA-BRCA 多组学数据", "#E8F1F8"), ("wave", "mRNA / CNV / miRNA", "#E8F1F8"), ("venn", "统一交集样本", "#E8F1F8")],
        ),
        (
            "2",
            "临床标签定义",
            "#348FB3",
            [("pam50", "PAM50 四分类", "#E9F4F7"), ("pulse", "生存事件 / 删失", "#FDF1E2"), ("clock", "生存时间", "#E9F4F7")],
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
            [("concat", "拼接 / 注意力", "#FAE8E8"), ("stack", "Stacking", "#FAE8E8"), ("gauge", "AUC / C-index", "#FAE8E8")],
        ),
    ]

    positions = [(60, 150), (520, 150), (980, 150), (60, 560), (520, 560), (980, 560)]

    svg = []
    svg.append('<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1040" viewBox="0 0 1500 1040" font-family="Microsoft YaHei, Arial, sans-serif">')
    svg.append("<defs>")
    svg.append('<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="6" stdDeviation="9" flood-color="#1E3A5F" flood-opacity="0.12"/></filter>')
    svg.append('<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#F8FBFD"/><stop offset="1" stop-color="#EDF3F8"/></linearGradient>')
    svg.append('<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#7D9CB7"/></marker>')
    svg.append("</defs>")
    svg.append('<rect width="1500" height="1040" fill="url(#bg)"/>')
    svg.append('<text x="750" y="52" text-anchor="middle" font-size="30" font-weight="700" fill="#173A5E">JSD-Guided Omics Imaging and Multi-Omics Fusion</text>')
    svg.append('<text x="750" y="86" text-anchor="middle" font-size="17" fill="#5E6E83">基于 JSD 引导组学图像化与多组学融合的乳腺癌分子分型及生存预测 · 方法概念图</text>')

    for i, (no, title, color, rows) in enumerate(cards_data):
        x, y = positions[i]
        svg.append(card(x, y, no, title, color, rows))

    # Row arrows
    svg.append(arrow(500, 330, 520, 330))
    svg.append(arrow(960, 330, 980, 330))
    svg.append(arrow(500, 740, 520, 740))
    svg.append(arrow(960, 740, 980, 740))

    # Phase separator between the two rows.
    svg.append('<rect x="60" y="522" width="1380" height="26" rx="13" fill="#EAF1F7" stroke="#D7E2EC" stroke-width="1.5"/>')
    svg.append('<text x="750" y="541" text-anchor="middle" font-size="14" font-weight="700" fill="#4C6A85">输入图像 → 建模 → 融合 → 评估</text>')

    svg.append('<rect x="540" y="970" width="420" height="44" rx="22" fill="#FBF0F0" stroke="#D9A3A3"/>')
    svg.append('<text x="750" y="999" text-anchor="middle" font-size="17" font-weight="700" fill="#A23E3E">下游可解释性：SHAP · Grad-CAM</text>')
    svg.append('<text x="750" y="1023" text-anchor="middle" font-size="13" fill="#6E7A8A">图 1. 研究方法概念图。特征筛选、标准化和模型拟合均在 5 折交叉验证训练折内完成。</text>')
    svg.append("</svg>")

    OUT.write_text("\n".join(svg), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
