#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a three-phase swim-lane conceptual technical route."""

from pathlib import Path


OUT = Path(r"E:\组学数据图像化\data\paper_figures\fig1_phase_route_v16_english.svg")


def icon(key: str, color: str) -> str:
    s = f'<g transform="translate(0,0)" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">'
    if key == "db":
        s += '<rect x="-28" y="-26" width="56" height="14" rx="6" fill="#E8F1F8"/><path d="M-28 -19 h56"/><rect x="-28" y="-6" width="56" height="14" rx="6" fill="#BBD5EA"/><path d="M-28 1 h56"/><rect x="-28" y="14" width="56" height="14" rx="6" fill="#D9EAF5"/>'
    elif key == "venn":
        s += '<circle cx="-8" cy="0" r="16"/><circle cx="8" cy="0" r="16"/>'
    elif key == "pam50":
        s += '<rect x="-20" y="-20" width="18" height="18" rx="3" fill="#6AA4C9"/><rect x="2" y="-20" width="18" height="18" rx="3" fill="#F2A45A"/><rect x="-20" y="2" width="18" height="18" rx="3" fill="#D86C6C"/><rect x="2" y="2" width="18" height="18" rx="3" fill="#81BBC8"/>'
    elif key == "variance":
        s += '<path d="M-26 20 v-40 M-9 20 v-26 M9 20 v-34 M26 20 v-16"/>'
    elif key == "nsre":
        s += '<path d="M-24 22 V-14 h11 v36 M-6 22 V4 h11 v18 M12 22 V-4 h11 v26" stroke-width="3"/>'
    elif key == "sort":
        s += '<path d="M-22 -18 v38 M-30 11 l8 9 8 -9 M22 -18 v38 M14 11 l8 9 8 -9"/>'
    elif key == "spiral":
        s += '<path d="M0 0 C5 -2 11 0 9 6 C7 11 0 13 -6 9 C-11 5 -11 -2 -6 -7 C0 -12 11 -11 13 -4" stroke-width="3"/>'
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


def wrap_text(text: str, max_chars: int):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if not current or len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def content_panel(x, y, color, items):
    blocks = []
    col_x = [20, 170, 320]
    for i, (key, title, desc) in enumerate(items):
        center = col_x[i] + 70
        title_lines = wrap_text(title, 14)
        desc_lines = wrap_text(desc, 18)
        title_svg = "".join(
            f'<text x="{center}" y="{154 if len(title_lines) == 1 else 170 - 16 + i * 20}" text-anchor="middle" font-size="14" font-weight="700" fill="#1E3A5F">{line}</text>'
            for i, line in enumerate(title_lines)
        )
        desc_svg = "".join(
            f'<text x="{center}" y="{202 if len(desc_lines) == 1 else 198 + i * 19}" text-anchor="middle" font-size="11.5" fill="#5E6E83">{line}</text>'
            for i, line in enumerate(desc_lines)
        )
        blocks.append(
            f'<g transform="translate({center},90)">{icon(key, color)}</g>'
            + title_svg
            + desc_svg
        )
    inner = (
        blocks[0]
        + '<line x1="160" y1="20" x2="160" y2="260" stroke="#E4EBF2" stroke-width="1.5"/>'
        + blocks[1]
        + '<line x1="310" y1="20" x2="310" y2="260" stroke="#E4EBF2" stroke-width="1.5"/>'
        + blocks[2]
    )
    return (
        f'<g transform="translate({x},{y})">'
        f'<rect x="0" y="0" width="480" height="280" rx="18" fill="#FFFFFF" stroke="#C9D9E8" stroke-width="1.5" filter="url(#shadow)"/>'
        + inner +
        "</g>"
    )


def phase_header(x, y, no, title, color):
    return (
        f'<g transform="translate({x},{y})">'
        f'<rect x="0" y="0" width="480" height="66" rx="18" fill="{color}"/>'
        f'<circle cx="34" cy="33" r="20" fill="#FFFFFF" opacity="0.95"/>'
        f'<text x="34" y="39" text-anchor="middle" font-size="16" font-weight="700" fill="{color}">{no}</text>'
        f'<text x="66" y="40" font-size="20" font-weight="700" fill="#FFFFFF">{title}</text>'
        "</g>"
    )


def shap_card(x, y):
    return (
        f'<g transform="translate({x},{y})">'
        '<rect x="0" y="0" width="730" height="150" rx="16" fill="#FFFFFF" stroke="#C9D9E8" stroke-width="1.5" filter="url(#shadow)"/>'
        '<g transform="translate(100,75)" stroke="#5E6E83" stroke-width="3">'
        '<line x1="0" y1="-34" x2="0" y2="34"/>'
        '<path d="M0 -34 L-28 -18 M0 -34 L28 -18" fill="none"/>'
        '<line x1="-28" y1="0" x2="0" y2="0" stroke="#2E6FA6"/>'
        '<line x1="0" y1="10" x2="34" y2="10" stroke="#D86C6C"/>'
        '<line x1="0" y1="24" x2="24" y2="24" stroke="#D86C6C"/>'
        '<line x1="-18" y1="-14" x2="0" y2="-14" stroke="#2E6FA6"/>'
        '</g>'
        '<text x="190" y="52" font-size="19" font-weight="700" fill="#1E3A5F">SHAP</text>'
        '<text x="190" y="82" font-size="13" fill="#5E6E83">Additive feature attribution</text>'
        '<text x="190" y="108" font-size="13" fill="#5E6E83">Blue = negative contribution; red = positive contribution</text>'
        "</g>"
    )


def gradcam_card(x, y):
    return (
        f'<g transform="translate({x},{y})">'
        '<rect x="0" y="0" width="730" height="150" rx="16" fill="#FFFFFF" stroke="#C9D9E8" stroke-width="1.5" filter="url(#shadow)"/>'
        '<g transform="translate(100,75)">'
        '<rect x="-34" y="-34" width="68" height="68" rx="6" fill="#EFF4F8" stroke="#5E6E83" stroke-width="2"/>'
        '<line x1="-34" y1="-12" x2="34" y2="-12" stroke="#C9D9E8"/>'
        '<line x1="-34" y1="10" x2="34" y2="10" stroke="#C9D9E8"/>'
        '<line x1="-12" y1="-34" x2="-12" y2="34" stroke="#C9D9E8"/>'
        '<line x1="12" y1="-34" x2="12" y2="34" stroke="#C9D9E8"/>'
        '<circle cx="4" cy="4" r="18" fill="#D86C6C" opacity="0.45"/>'
        '</g>'
        '<text x="190" y="52" font-size="19" font-weight="700" fill="#1E3A5F">Grad-CAM</text>'
        '<text x="190" y="82" font-size="13" fill="#5E6E83">Gradient-weighted class activation mapping</text>'
        '<text x="190" y="108" font-size="13" fill="#5E6E83">Heat map highlights image regions driving the prediction</text>'
        "</g>"
    )


def arrow(x0, y0, x1, y1):
    return f'<path d="M{x0} {y0} L{x1} {y1}" stroke="#7D9CB7" stroke-width="3" fill="none" marker-end="url(#arrow)"/>'


def main():
    phases = [
        (
            "I",
            "Data Preparation",
            "#2E6FA6",
            [
                ("db", "Data acquisition", "TCGA-BRCA", "#E8F1F8"),
                ("venn", "Sample alignment", "mRNA / CNV / miRNA", "#E8F1F8"),
                ("pam50", "Label definition", "PAM50 / survival", "#E8F1F8"),
            ],
        ),
        (
            "II",
            "Feature Engineering & Imaging",
            "#60A5C7",
            [
                ("variance", "Feature selection", "Low-variance / F / L1", "#EAF4F8"),
                ("nsre", "NSRE ranking", "Class discrimination", "#EAF4F8"),
                ("spiral", "Spiral imaging", "Gray / functional maps", "#EAF4F8"),
            ],
        ),
        (
            "III",
            "Modeling, Fusion & Evaluation",
            "#D86C6C",
            [
                ("ml", "Single-omics modeling", "Classical ML / MLP", "#FDF1E2"),
                ("concat", "Multi-omics fusion", "Concat / attention", "#FAE8E8"),
                ("gauge", "Cross-validation", "AUC / C-index", "#FAE8E8"),
            ],
        ),
    ]

    cols = [40, 540, 1040]
    y_top = 100

    svg = []
    svg.append('<svg xmlns="http://www.w3.org/2000/svg" width="1560" height="620" viewBox="0 0 1560 620" font-family="Microsoft YaHei, Arial, sans-serif">')
    svg.append("<defs>")
    svg.append('<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="6" stdDeviation="9" flood-color="#1E3A5F" flood-opacity="0.12"/></filter>')
    svg.append('<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#F8FBFD"/><stop offset="1" stop-color="#EDF3F8"/></linearGradient>')
    svg.append('<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#7D9CB7"/></marker>')
    svg.append("</defs>")
    svg.append('<rect width="1560" height="620" fill="url(#bg)"/>')
    svg.append('<text x="780" y="36" text-anchor="middle" font-size="28" font-weight="700" fill="#173A5E">NSRE-Guided Omics Imaging and Multi-Omics Fusion</text>')
    svg.append('<text x="780" y="62" text-anchor="middle" font-size="16" fill="#5E6E83">A method overview for breast cancer molecular subtyping and survival prediction</text>')

    for i, (no, title, color, subs) in enumerate(phases):
        x = cols[i]
        # Subtle phase background band that groups each column.
        svg.append(f'<rect x="{x - 12}" y="{y_top - 12}" width="504" height="382" rx="24" fill="#F2F6FA" stroke="#DDE7F0" stroke-width="1.5"/>')
        svg.append(phase_header(x, y_top, no, title, color))
        items = [(key, sub_title, desc) for key, sub_title, desc, light in subs]
        svg.append(content_panel(x, y_top + 78, color, items))

    # Horizontal phase arrows.
    svg.append(arrow(520, 133, 540, 133))
    svg.append(arrow(1020, 133, 1040, 133))

    # Vertical arrows inside each column between sub-cards.
    for x in cols:
        cx = x + 240
        svg.append(arrow(cx, y_top + 66, cx, y_top + 78))

    svg.append('<text x="780" y="492" text-anchor="middle" font-size="15" font-weight="700" fill="#6E7A8A">Downstream interpretability</text>')
    svg.append('<rect x="590" y="505" width="180" height="40" rx="20" fill="#FBF0F0" stroke="#D9A3A3"/>')
    svg.append('<text x="680" y="531" text-anchor="middle" font-size="16" font-weight="700" fill="#A23E3E">SHAP</text>')
    svg.append('<rect x="790" y="505" width="180" height="40" rx="20" fill="#FBF0F0" stroke="#D9A3A3"/>')
    svg.append('<text x="880" y="531" text-anchor="middle" font-size="16" font-weight="700" fill="#A23E3E">Grad-CAM</text>')
    svg.append('<text x="780" y="586" text-anchor="middle" font-size="13" fill="#6E7A8A">Figure 1. Method overview. Feature selection, scaling, and model fitting are performed within each training fold.</text>')
    svg.append("</svg>")

    OUT.write_text("\n".join(svg), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
