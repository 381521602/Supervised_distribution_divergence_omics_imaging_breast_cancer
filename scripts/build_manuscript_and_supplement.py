#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the Chinese manuscript and detailed supplementary Word documents."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = DATA / "paper_figures"
OUT_MAIN = ROOT / "Supervised_distribution_divergence_guided_omics_imaging_论文中文稿.docx"
OUT_SUPP = ROOT / "Supervised_distribution_divergence_guided_omics_imaging_补充材料.docx"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(
    0,
    str(
        Path(
            r"C:\Users\38152\.codex\plugins\cache\openai-primary-runtime\documents\26.819.11345\skills\documents\scripts"
        )
    ),
)
from table_geometry import apply_table_geometry, column_widths_from_weights  # noqa: E402


BLUE = RGBColor(0x2E, 0x74, 0xB5)
DARK = RGBColor(0x1F, 0x4D, 0x78)
INK = RGBColor(0x0B, 0x25, 0x45)
MUTED = RGBColor(0x55, 0x55, 0x55)
HEADER_FILL = "F4F6F9"
HEADER_FILL_2 = "E8EEF5"


def set_run_font(run, latin="Times New Roman", east="宋体", size=None, bold=None, color=None):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east)
    run._element.get_or_add_rPr().rFonts.set(qn("w:cs"), latin)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def add_paragraph(
    doc,
    text,
    *,
    style=None,
    align=None,
    size=11,
    bold=False,
    italic=False,
    color=None,
    latin="Times New Roman",
    east="宋体",
    before=None,
    after=None,
    line=None,
):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    set_run_font(run, latin=latin, east=east, size=size, bold=bold, color=color)
    if italic:
        run.font.italic = True
    pf = p.paragraph_format
    if before is not None:
        pf.space_before = Pt(before)
    if after is not None:
        pf.space_after = Pt(after)
    if line is not None:
        pf.line_spacing = line
    return p


def add_heading(doc, text, level=1):
    p = doc.add_heading(level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    sizes = {1: 16, 2: 13, 3: 12}
    colors = {1: BLUE, 2: BLUE, 3: DARK}
    before = {1: 16, 2: 12, 3: 8}
    after = {1: 7, 2: 5, 3: 4}
    run = p.add_run(text)
    set_run_font(
        run,
        latin="Times New Roman",
        east="微软雅黑",
        size=sizes[level],
        bold=True,
        color=colors[level],
    )
    p.paragraph_format.space_before = Pt(before[level])
    p.paragraph_format.space_after = Pt(after[level])
    return p


def add_body(doc, text, justify=True, after=6, size=11, first_indent=True):
    p = add_paragraph(
        doc,
        text,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY if justify else WD_ALIGN_PARAGRAPH.LEFT,
        size=size,
        after=after,
        line=1.25,
    )
    if first_indent and not text.lstrip().startswith("["):
        p.paragraph_format.first_line_indent = Pt(size * 2)
    return p


def add_formula(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_run_font(run, latin="Cambria Math", east="宋体", size=10.5)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    return p


def add_numbered_formula(doc, text, number):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.tab_stops.add_tab_stop(Inches(6.4), WD_TAB_ALIGNMENT.RIGHT)
    run = p.add_run(text + f"\t({number})")
    set_run_font(run, latin="Cambria Math", east="宋体", size=10.5)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    return p


def add_bullet(doc, text, after=3):
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    set_run_font(run, size=10.5)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.15
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    if text.startswith("表") or text.startswith("Table"):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        set_run_font(run, size=9.5, bold=True, color=DARK, east="宋体")
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        body = doc.element.body
        tables = body.findall(qn("w:tbl"))
        if tables:
            tables[-1].addprevious(p._p)
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        set_run_font(run, size=9, color=MUTED, east="宋体")
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(9)
    return p


def add_figure(doc, path, caption, width=6.2):
    if not Path(path).exists():
        add_body(doc, f"[缺失图片: {path}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    add_caption(doc, caption)


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_text(cell, text, *, bold=False, size=8.5, align=None, east="宋体"):
    cell.text = ""
    p = cell.paragraphs[0]
    if align is not None:
        p.alignment = align
    run = p.add_run(str(text))
    set_run_font(run, size=size, bold=bold, east=east)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.space_before = Pt(1)


def apply_three_line_table(table):
    tbl = table._tbl
    tbl_pr = tbl.tblPr

    old_borders = tbl_pr.find(qn("w:tblBorders"))
    if old_borders is not None:
        tbl_pr.remove(old_borders)

    borders = OxmlElement("w:tblBorders")

    def add_border(tag, val, sz=None):
        edge = OxmlElement(tag)
        edge.set(qn("w:val"), val)
        if sz is not None:
            edge.set(qn("w:sz"), str(sz))
            edge.set(qn("w:space"), "0")
            edge.set(qn("w:color"), "000000")
        borders.append(edge)

    add_border("w:top", "single", 12)
    add_border("w:bottom", "single", 12)
    add_border("w:left", "none")
    add_border("w:right", "none")
    add_border("w:insideH", "none")
    add_border("w:insideV", "none")
    tbl_pr.append(borders)

    # Explicitly clear all cell-level borders, then set the middle line only under
    # the header row. This overrides any borders inherited from the default table
    # style.
    for row in table.rows:
        for cell in row.cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            shd = tc_pr.find(qn("w:shd"))
            if shd is not None:
                tc_pr.remove(shd)

            old_tc_borders = tc_pr.find(qn("w:tcBorders"))
            if old_tc_borders is not None:
                tc_pr.remove(old_tc_borders)

            tc_borders = OxmlElement("w:tcBorders")
            for tag in ("w:top", "w:bottom", "w:left", "w:right", "w:insideH", "w:insideV"):
                edge = OxmlElement(tag)
                edge.set(qn("w:val"), "none")
                tc_borders.append(edge)
            tc_pr.append(tc_borders)

    for cell in table.rows[0].cells:
        tc_pr = cell._tc.get_or_add_tcPr()
        tc_borders = tc_pr.find(qn("w:tcBorders"))
        if tc_borders is None:
            tc_borders = OxmlElement("w:tcBorders")
            tc_pr.append(tc_borders)

        top = tc_borders.find(qn("w:top"))
        if top is None:
            top = OxmlElement("w:top")
            tc_borders.append(top)
        top.set(qn("w:val"), "single")
        top.set(qn("w:sz"), "16")
        top.set(qn("w:space"), "0")
        top.set(qn("w:color"), "000000")

        bottom = tc_borders.find(qn("w:bottom"))
        if bottom is None:
            bottom = OxmlElement("w:bottom")
            tc_borders.append(bottom)
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "8")
        bottom.set(qn("w:space"), "0")
        bottom.set(qn("w:color"), "000000")

    for cell in table.rows[-1].cells:
        tc_pr = cell._tc.get_or_add_tcPr()
        tc_borders = tc_pr.find(qn("w:tcBorders"))
        if tc_borders is None:
            tc_borders = OxmlElement("w:tcBorders")
            tc_pr.append(tc_borders)
        bottom = tc_borders.find(qn("w:bottom"))
        if bottom is None:
            bottom = OxmlElement("w:bottom")
            tc_borders.append(bottom)
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "16")
        bottom.set(qn("w:space"), "0")
        bottom.set(qn("w:color"), "000000")


def add_dataframe_table(doc, df, weights, header_fill=None, font_size=8.5):
    headers = list(df.columns)
    table = doc.add_table(rows=1, cols=len(headers))
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        set_cell_text(hdr[i], h, bold=True, size=font_size, east="微软雅黑")
        if header_fill:
            shade_cell(hdr[i], header_fill)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, h in enumerate(headers):
            val = row[h]
            if isinstance(val, (np.floating, float)):
                txt = f"{val:.4f}"
            else:
                txt = str(val)
            set_cell_text(cells[i], txt, size=font_size)
    widths = column_widths_from_weights(weights)
    apply_table_geometry(table, widths)
    apply_three_line_table(table)
    # small spacing after table
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.space_before = Pt(0)
    run = p.add_run("")
    set_run_font(run, size=2)
    return table


def read_tsv(name):
    return pd.read_csv(DATA / name, sep="\t")


def setup_document():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(1)
    sec.bottom_margin = Inches(1)
    sec.left_margin = Inches(1)
    sec.right_margin = Inches(1)
    sec.header_distance = Inches(0.492)
    sec.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal_rfonts = normal._element.get_or_add_rPr().rFonts
    normal_rfonts.set(qn("w:ascii"), "Times New Roman")
    normal_rfonts.set(qn("w:hAnsi"), "Times New Roman")
    normal_rfonts.set(qn("w:cs"), "Times New Roman")
    normal_rfonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    for level in [1, 2, 3]:
        s = styles[f"Heading {level}"]
        s.font.name = "Times New Roman"
        heading_rfonts = s._element.get_or_add_rPr().rFonts
        heading_rfonts.set(qn("w:ascii"), "Times New Roman")
        heading_rfonts.set(qn("w:hAnsi"), "Times New Roman")
        heading_rfonts.set(qn("w:cs"), "Times New Roman")
        heading_rfonts.set(qn("w:eastAsia"), "微软雅黑")

    footer = sec.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run("Distribution-divergence-guided omics imaging · Page ")
    set_run_font(run, size=8, color=MUTED, east="宋体")
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    fp._p.append(fld)
    return doc


def add_metrics_note(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(
        "说明：所有表格与图中结果均为“基于训练折内筛选特征的结果”，在 5 折交叉验证下给出均值±标准差。"
        "PAM50 四分类报告 Accuracy 和 Macro-F1；生存预测报告 ROC AUC 和 Cox C-index。"
        "ROC AUC 以模型输出的死亡风险概率/风险分数作为连续预测分数计算；C-index 使用同一风险分数、生存时间与事件状态，通过 lifelines.utils.concordance_index 计算。"
    )
    set_run_font(run, size=9, color=MUTED, east="宋体")
    p.paragraph_format.space_after = Pt(8)
    return p


def build_main():
    doc = setup_document()

    # Title block
    add_paragraph(
        doc,
        "Supervised distribution-divergence-guided omics imaging and multi-omics fusion for breast cancer molecular subtyping and survival prediction",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        size=15,
        bold=True,
        color=INK,
        east="微软雅黑",
        after=4,
    )
    add_paragraph(
        doc,
        "监督式分布散度引导的组学图像化与多组学融合用于乳腺癌分子分型及生存预测",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        size=12,
        bold=True,
        color=DARK,
        after=8,
    )
    add_paragraph(
        doc,
        "中文初稿 · Original Research 准备稿 · 与 Frontiers Research Topic “Molecular Stress Responses and Adaptive Reprogramming in Disease Progression and Therapy” 对齐",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        size=9,
        color=MUTED,
        after=12,
    )

    add_heading(doc, "摘要", 1)
    add_body(doc, "背景：乳腺癌的高度异质性使单一组学难以稳定刻画 PAM50 分子亚型和总生存风险。多组学整合具有理论上的互补优势，但组学数据存在特征维度高、样本量有限、分布尺度不一致以及特征之间缺乏自然二维邻接关系等问题。", size=10.5)
    add_body(doc, "方法：本研究以 TCGA-BRCA 的 mRNA、CNV 和 miRNA 为对象，提出一种监督式分布散度引导的组学空间表示框架（supervised information-divergence-guided omics spatial representation framework），即以 Jensen–Shannon divergence（JSD；实现中简称 NSRE）评估每个特征的判别重要性，并将其编码为可解释的空间表示。所有特征筛选、标准化和模型拟合均在 5 折交叉验证训练折内完成；按 JSD/NSRE 重要性降序并将高分特征置于中心，采用中心向外螺旋填充形成单通道灰度图，再输入全尺寸卷积神经网络（FullSizeCNN）。同时比较 LogisticRegression、RandomForest、GradientBoosting、SVC、KNN、MLP、CoxPH 及多种多组学融合策略。", size=10.5)
    add_body(doc, "结果：在单组学 PAM50 四分类中，mRNA LogisticRegression 的 Accuracy 为 0.9628±0.0167、Macro-F1 为 0.9575±0.0199；CNV 最优 MLP 为 0.7213±0.0156；miRNA 最优 LogisticRegression 为 0.8410±0.0120。生存预测中，mRNA FullSizeCNN 的 C-index 为 0.7125±0.0303，CNV CoxPH 为 0.7109±0.0339，miRNA FullSizeCNN 为 0.6067±0.0703。多组学整合中，PAM50 的 Transformer 达到 Accuracy 0.9270±0.0132、Macro-F1 0.9288±0.0128；生存预测的三组学 Concat MLP 达到 C-index 0.7588±0.0492，CNN 晚期平均融合达到 ROC AUC 0.7591±0.0619。配对检验显示，上述最优融合均未较对应最优单组学形成统计显著优势（所有配对检验 p≥0.093）。可解释分析识别出 ESR1、TFF1、AGR3、FOXC1、MIA、FABP7、CCL19、MS4A1 等关键因子，富集于雌激素信号、细胞增殖调控、细胞分裂和上皮发育等通路。方案2功能类别遮盖分析在控制特征集合大小后，未能建立稳健的类别特异效应（masking analyses did not establish robust category-specific effects after controlling for feature-set size）。", size=10.5)
    add_body(doc, "结论：监督式分布散度引导的组学空间表示框架能够将高维组学特征转换为适合全尺寸卷积建模的图像表示，并支持关键因子可视化；该表示的价值在于可解释性而非预测精度提升。mRNA 是 PAM50 分型重建的主要信号，生存预测更依赖多组学互补；当前整合策略的增益有限，尚需外部验证和更稳健的融合设计。", size=10.5)
    add_body(doc, "关键词：乳腺癌；多组学；组学图像化；Jensen–Shannon divergence；全尺寸卷积；PAM50；生存预测", size=10.5)

    add_heading(doc, "1. 引言", 1)
    add_body(
        doc,
        "乳腺癌是全球女性最常见的恶性肿瘤之一，其发生、进展和治疗反应受遗传、转录、表观遗传及微环境等多层因素调控，分子与临床异质性极高[1-3]。基于基因表达的固有分子分型（intrinsic subtypes）将乳腺癌划分为 Luminal A、Luminal B、Basal-like、HER2-enriched 和 Normal-like 等亚型，其中 PAM50 基因集已成为分子分型和复发风险估计的重要工具[3]。大型基因组研究进一步表明，乳腺癌的分子景观同时受细胞突变、拷贝数变异（copy number variation, CNV）、转录组和 miRNA 等事件共同塑造[4-5]；三阴性乳腺癌、浸润性小叶癌等亚型内部也存在可重复的分子亚群[6-7]。总生存期（overall survival, OS）直接反映患者长期结局，因此从多组学层面同时刻画 PAM50 亚型和生存风险具有明确的临床转化价值。",
    )
    add_body(
        doc,
        "The Cancer Genome Atlas（TCGA）及其泛癌计划（Pan-Cancer Analysis Project）提供了大规模、多平台、公开可及的肿瘤分子数据，使系统比较乳腺癌及其他癌种的跨组学分子模式成为可能[4,8]。多平台分析显示，组织起源和分子亚型共同决定肿瘤的全局分子特征，单一组学往往只能捕捉部分信号[9]。多组学数据整合被普遍认为能够融合互补信息，并已在癌症分类、预后建模和药物反应预测等任务中取得进展[10-11]。然而，多组学整合仍然面临特征维度高、有效样本量有限、不同组学分布尺度不一致、数据缺失和批间差异等挑战，直接构建可解释且可重复的预测模型并不容易。",
    )
    add_body(
        doc,
        "在高维组学建模中，特征选择是控制过拟合、降低计算成本和增强生物学可解释性的关键步骤[12]。常用方法包括基于方差或单变量统计量的过滤式方法、基于 LASSO/弹性网络的嵌入式稀疏方法、基于互信息的最大相关最小冗余（mRMR）方法，以及 Boruta 等基于随机森林重要性的包装式方法[13-17]。这些方法分别从稳定性、线性判别、信息冗余和扰动重要性等角度对特征进行筛选，但不同方法的适用条件与最终保留特征集合并不完全一致。",
    )
    add_body(
        doc,
        "在传统机器学习层面，逻辑回归、支持向量机、随机森林、梯度提升树和 k 近邻等算法已在基因组与临床预测中得到广泛应用[18-20]，并因其可解释性、计算效率和对表格型数据较好的稳健性而长期作为基线[21-23]。这类方法通常直接以基因表达或拷贝数向量为输入，因而无法自然利用特征之间可能存在的空间或网络邻接关系。类似地，生态预测领域从统计模型、经典机器学习到深度学习的方法演进也表明，在数据稀缺、时空异质性和可解释性约束下，模型选择需要同时兼顾数据结构和任务目标[24]。",
    )
    add_body(
        doc,
        "深度学习的进展为生物医学数据分析提供了新的表征学习能力[21-23]。卷积神经网络（CNN）在图像识别中表现出强大的局部特征提取能力，从 LeNet、AlexNet 到 ResNet 和 Squeeze-and-Excitation Network 等结构不断提升了模型容量与泛化能力[25-28]；U-Net 等编码器-解码器结构则广泛应用于生物医学图像[29]。Transformer 与注意力机制进一步将序列和集合型数据的建模能力扩展到多模态场景[30]。针对序列型生物数据，混合膨胀卷积、门控卷积和深度可分离卷积的结构也被用于单碱基分辨率的核小体占据率预测，说明按数据尺度与感受野定制卷积模块有助于增强序列建模能力[31]。然而，深度学习模型参数多、样本需求大，若缺乏合适的输入表示和严格的防泄漏验证，容易在小样本组学任务上过拟合。",
    )
    add_body(
        doc,
        "组学数据的“图像化”是连接 CNN 与高维分子数据的重要途径。DeepInsight 通过把非图像数据映射为二维图像，使 CNN 能够处理基因表达等表格数据[32]；DeepInsight-3D 将该思想扩展至多组学药物反应预测[33]。近期研究也利用转录组特征图进行癌症类型和生存预测[34]。这些方法说明，只要特征映射方式合理，图像化可以在保留原始数值的同时引入空间归纳偏置；但组学特征之间通常并不存在像素意义上的自然邻接关系，排序和填充方式会直接影响二维位置所表达的生物学含义。",
    )
    add_body(
        doc,
        "生存分析是临床预后建模的核心。Cox 比例风险模型提供了半参数风险建模框架[35]，C-index 是评估生存风险排序一致性的常用指标[36]，而 Uno 等提出的 C 统计量则进一步考虑了删失和风险预测的整体适用性[37]。随机生存森林和 DeepSurv 将集成学习与深度神经网络扩展到生存数据[38-39]。在乳腺癌生存预测中，已有工作利用多组学决策级融合构建总体生存预测流程[40]。",
    )
    add_body(
        doc,
        "多组学融合策略从早期特征拼接、决策级融合和 Stacking，逐渐发展到基于注意力、低秩双线性融合和图神经网络的深度整合[10-11,40]。同时，生成对抗网络（GAN）及其 Wasserstein 变体为小样本组学数据扩增提供了条件生成方案，但必须严格控制测试集信息泄漏[41-43]。在这些融合与增强策略中，如何在不同组学之间保持样本一致、特征尺度统一和模型可解释，仍缺乏系统比较。",
    )
    add_body(
        doc,
        "可解释性是组学预测模型向临床转化的重要前提。saliency map、Grad-CAM、LIME 和 SHAP 等方法从梯度、局部扰动或博弈论角度解释模型输出[45-48]。对于图像化组学模型，可解释性不仅要求给出重要基因，还要求模型关注的像素位置能够反推回对应的生物学特征；这进一步凸显了图像化排序方式的重要性。",
    )
    add_body(
        doc,
        "本研究针对上述问题，提出以 Jensen–Shannon divergence（JSD；实现中简称 NSRE）引导的组学图像化框架：先评估每个基因或探针对结局类别的区分能力，再按 JSD/NSRE 重要性降序并以中心向外螺旋方式填充图像，使高重要性特征集中于图像中心；随后以全尺寸卷积神经网络（FullSizeCNN）作为图像化基线，并与多种传统机器学习、MLP 和经典 CNN 结构进行对比。该方案既保留原始表达值，又将判别重要性显式编码到二维空间，便于后续关键因子定位。",
    )
    add_body(
        doc,
        "本文提出三个假设：（1）NSRE 排序比随机排序或平均表达排序更适合 PAM50 四分类和生存预测；（2）在小样本组学图像上，FullSizeCNN 能够提供与传统机器学习基线可比或互补的预测性能；（3）在统一交集样本上，多组学融合可较最优单组学带来生存预测上的稳健增益，但在 PAM50 中增益有限。",
    )
    add_body(
        doc,
        "本文的贡献包括：提出 NSRE 引导的中心-螺旋组学图像化框架；系统比较传统机器学习、MLP、FullSizeCNN 及多种多组学融合策略；在严格 5 折交叉验证下报告均值和标准差；利用 SHAP、CNN saliency 和基因功能类别图开展关键因子可视化；并通过方案2功能类别遮盖实验与等量随机遮盖对照，评估不同功能模块对 PAM50 分类和生存预测的贡献。",
    )

    add_heading(doc, "2. 材料与方法", 1)
    add_body(doc, "本部分仅给出核心方法与参数。具体样本清单见补充材料表 S1；特征筛选算法组合见表 S2，最终特征数量见表 S3；单组学全部结果见表 S4，交集单组学基线见表 S5；三组学融合结果见表 S6，两两组学融合结果见表 S7，Stacking 与复杂方法结果见表 S8–S9；融合统计检验见表 S10；图像化结构对比见表 S11–S13；全尺寸卷积 Dense 等价基线见表 S14 与图 S3；随机排列排序对照见表 S15 与图 S4；可解释性补充图见图 S5–S13；方案2功能类别遮盖与统计检验见表 S16–S21；应激与适应性重编程通路分析见表 S22 与图 S14；PAM50 50 基因剔除敏感性分析见表 S23 与图 S15。")
    add_figure(doc, FIG / "fig1_route_v25.png", "Figure 1. Overall technical route. The three phases are data preparation, feature engineering, and modeling/fusion/evaluation; SHAP, Grad-CAM/saliency, and functional masking are used as downstream interpretability modules.", width=6.8)
    add_heading(doc, "2.1 数据来源与样本对齐", 2)
    add_body(
        doc,
        "本研究使用公开数据库 TCGA-BRCA 数据，来自 Genomic Data Commons（GDC）和 UCSC Xena。mRNA 采用 TCGA BRCA HiSeqV2 的 log2(RSEM+1) 基因表达矩阵；CNV 采用 GISTIC2 基因级阈值化拷贝数矩阵；miRNA 采用 miRNA 表达矩阵。临床标签包括 PAM50 亚型、生存状态、生存时间等。按照 TCGA 患者条码提取原发肿瘤样本，并对三组学样本求交集，优先选择每个病例的 -01 原发肿瘤样本。所有数据为公开、去标识化数据，不涉及新的患者样本采集；各组学与交集样本量见补充材料表 S1。",
    )
    add_body(
        doc,
        "PAM50 任务在剔除 Normal-like 后保留四个主要亚型：mRNA 833 例、CNV 818 例、miRNA 497 例，三组学交集 493 例。生存任务分别有 mRNA 1073 例、CNV 1057 例、miRNA 739 例，三组学交集 731 例。两两交集样本量分别为：PAM50 mRNA∩CNV 818、mRNA∩miRNA 497、CNV∩miRNA 493；生存 mRNA∩CNV 1055、mRNA∩miRNA 737、CNV∩miRNA 733。",
    )
    add_dataframe_table(
        doc,
        pd.DataFrame(
            {
                "Task": ["PAM50", "Survival"],
                "mRNA": ["833", "1073"],
                "CNV": ["818", "1057"],
                "miRNA": ["497", "739"],
                "Triple-omics intersection": ["493", "731"],
            }
        ),
        [1.4, 1.0, 1.0, 1.0, 1.4],
    )
    add_caption(doc, "Table 1. Sample sizes by omics and triple-omics intersection")

    add_heading(doc, "2.2 标签定义", 2)
    add_body(
        doc,
        "PAM50 分子分型为四分类标签，类别为 Luminal A、Luminal B、Basal-like、HER2-enriched。生存分析以 OS 事件为二分类标签（1=死亡，0=删失），以诊断至死亡或末次随访时间作为生存时间。由于死亡事件在数据集中比例较低，生存任务存在类别不平衡。",
    )
    add_body(
        doc,
        "需要明确，除 CoxPH 与 DeepSurv 属于时间到事件模型外，其余模型（LogisticRegression、SVC、RandomForest、MLP、FullSizeCNN 等）以二元交叉熵拟合 OS 事件，本质上是死亡状态分类（mortality-status classification）；其输出被作为连续风险分数，再结合生存时间计算 C-index 进行生存排序评估（survival-ranking evaluation），因此本文所称“生存预测”并不等同于严格的风险率建模。",
    )

    add_heading(doc, "2.3 特征筛选与 JSD/NSRE", 2)
    add_body(
        doc,
        "为避免测试集信息泄漏，所有特征筛选均在每个交叉验证训练折内部完成。基础特征筛选方法包括低方差过滤、F 值（ANOVA）、L1/LASSO 和 Jensen–Shannon divergence（JSD；实现中简称 NSRE）[44]。JSD 通过比较不同类别中特征值的分箱直方图，计算两个离散分布之间的对称信息散度。",
    )
    add_body(doc, "低方差过滤首先移除训练折内方差为零或近似常数的特征；F 值采用单因素 ANOVA 比较连续特征在不同类别间均值差异；L1/LASSO 通过线性支持向量机的稀疏系数进行嵌入式选择。三种方法分别从稳定性、线性判别和稀疏正则化角度降低维度。特征筛选算法组合方式见补充材料表 S2。")
    add_numbered_formula(doc, "Var(x_j) = (1/N) ∑_i (x_ij - x̄_j)²;  F_j = MS_between / MS_within;  min_w (1/2) ||Xw - y||² + λ ||w||₁", 1)
    add_body(
        doc,
        "JSD 计算式：对类别 c 和 d 的特征直方图 P、Q，定义 D_JSD(P||Q) = 0.5 ∑_x [ P(x) log_2( 2P(x) / (P(x)+Q(x)) ) + Q(x) log_2( 2Q(x) / (P(x)+Q(x)) ) ]。二分类时直接计算两类间 JSD。多分类策略与具体组学相关：PAM50 mRNA 和生存数据使用平均 pairwise JSD，PAM50 CNV 使用 one-vs-rest max JSD，PAM50 miRNA 先使用 pairwise JSD 粗筛、再以 L1 精筛。为保证数值稳定，对直方图加入极小常数 ε 并进行概率归一化。",
    )
    add_numbered_formula(doc, "D_JSD(P||Q) = 0.5 ∑_x [ P(x) log_2( 2P(x) / (P(x)+Q(x)) ) + Q(x) log_2( 2Q(x) / (P(x)+Q(x)) ) ]", 2)
    add_body(
        doc,
        "特征数量通过 200/50/200 基础数量及 2×、3×、4×、5× 扩增进行系统比较。最终 PAM50 数据集特征数分别为 mRNA 400、CNV 50、miRNA 600；生存数据集特征数分别为 mRNA 200、CNV 150、miRNA 600。选择依据为在 Accuracy/Macro-F1 或 ROC AUC/C-index 接近最优时，优先选择计算效率更高、稳定性更好的特征数；最终特征数量与图像尺寸对应关系见补充材料表 S3。",
    )
    add_body(
        doc,
        "需要说明的是，特征数量的选择与最终性能评估共用同一 5 折交叉验证，因此本文报告的 Accuracy、Macro-F1、ROC AUC 与 C-index 属于探索性交叉验证估计（exploratory CV estimates），可能存在对最优特征数量的选择偏倚。受限于样本量，本文未另行构造嵌套交叉验证或独立验证集；最终特征数量应视为便于后续统一建模与比较的工作性设定，而非经过严格无偏验证的超参数。",
    )

    add_heading(doc, "2.4 组学图像化", 2)
    add_body(
        doc,
        "对每个任务和组学，先计算每个特征的 JSD/NSRE 分数并按降序排序。采用中心向外的螺旋顺序，将最高重要性特征放在图像中心，次高重要性特征沿螺旋向外排列。若特征数小于网格容量，剩余位置补零。图像为单通道灰度图，像素值即对应组学特征值；模型训练时仅在训练折内估计标准化参数并应用于训练折与测试折。最终图像尺寸为：PAM50 mRNA 20×20、CNV 8×8、miRNA 25×25；生存 mRNA 15×15、CNV 13×13、miRNA 25×25。",
    )
    add_body(
        doc,
        "为增强像素邻接的生物学可解释性，进一步采用方案2：先按功能大类将特征分块，再在块内按 JSD/NSRE 降序，并按块平均 JSD/NSRE 由中心向外螺旋填充。mRNA 和 CNV 的功能类别基于基因功能注释，miRNA 先由 MIMAT 映射至 hsa-miR 名称，再按 miRNA 家族/已知功能归类。对应生成功能类别图；在遮盖实验中，将目标功能类别的像素区域置零后重新训练 FullSizeCNN，并与不遮盖基线及同数量随机基因遮盖对照比较。",
    )
    add_dataframe_table(
        doc,
        pd.DataFrame(
            {
                "Task": ["PAM50", "PAM50", "PAM50", "Survival", "Survival", "Survival"],
                "Omics": ["mRNA", "CNV", "miRNA", "mRNA", "CNV", "miRNA"],
                "Final features": ["400", "50", "600", "200", "150", "600"],
                "Image size": ["20×20", "8×8", "25×25", "15×15", "13×13", "25×25"],
                "Zero-padded positions": ["0", "14", "25", "25", "19", "25"],
            }
        ),
        [1.0, 0.8, 0.9, 1.0, 0.8],
    )
    add_caption(doc, "Table 2. Final feature counts and NSRE image sizes")

    add_heading(doc, "2.5 模型", 2)
    add_body(
        doc,
        "传统机器学习基线包括 LogisticRegression、RandomForest、GradientBoosting、线性 SVC、KNN 和 MLP；生存预测另纳入 CoxPH。LogisticRegression 使用 L2 默认正则并增加最大迭代；RandomForest 使用 200 棵树；GradientBoosting 使用 100 轮；SVC 使用线性核；KNN 使用 k=5；MLP 采用 128-64 隐层和 Dropout。CoxPH 采用 top-100 方差特征和 0.1 惩罚。",
    )
    add_body(doc, "MLP 与 FullSizeCNN 均使用 Adam 优化器，学习率 1×10⁻³、权重衰减 1×10⁻⁴，训练 30 个 epoch，批大小为 64。PAM50 使用交叉熵损失，输出四类概率；生存预测使用二元交叉熵损失，输出死亡风险概率。训练前仅在训练折上估计 StandardScaler 参数，测试折使用同一 scaler 变换。")
    add_body(
        doc,
        "FullSizeCNN 的输入为 H×W×1 图像，第一层使用与 H×W 相同的全尺寸卷积核，输出 32 个 1×1 特征图；随后展平为 32 维向量，经 64 维全连接、ReLU 和 Dropout 后输出分类 logits 或生存风险。该结构参数较少，与图像尺寸解耦，适合当前小样本高维组学图像。",
    )
    add_figure(doc, FIG / "fig2_fullsize_cnn_v4.png", "Figure 2. FullSizeCNN architecture. An H×W×1 NSRE image is transformed by a full-size convolutional layer with H×W kernels and 32 filters, flattened to 32 features, and passed through a 64-unit ReLU/Dropout MLP head. PAM50 uses H=W=20 and four output units; survival uses H=W=15 and one risk output.", width=6.2)
    add_body(
        doc,
        "多组学整合采用三类策略：（1）特征级简单拼接后接传统 ML/MLP；（2）每个组学独立 FullSizeCNN 分支后进行特征拼接、门控注意力融合或概率晚期平均融合；（3）9 模型 Stacking、Transformer 跨组学注意力、DeepSurv、低秩双线性融合、图神经网络和多任务学习等复杂策略。",
    )
    add_body(
        doc,
        "复杂方法的关键超参数固定如下：Transformer 采用 d_model=32、nhead=4、layers=2、dim_feedforward=64、dropout=0.1，三组学分别投影为 32 维 token 后取均值池化；低秩双线性融合采用秩 r=8、投影维度 d=32；GNN 以基因间绝对 Pearson 相关系数构建邻接矩阵，保留每个节点的 top-k=40 邻居并加自环后对称归一化，使用两层 GCN（1→32→32）；9 模型 Stacking 以内部 5 折分层 CV 生成各基模型 out-of-fold 预测，meta-learner 为 LogisticRegression（max_iter=3000），外评估仍为 5 折分层 CV；WGAN-GP 的生成器输入噪声维度 64、类别嵌入维度 16，判别器每次迭代 3 次、梯度惩罚权重 λ_GP=10；DeepSurv 使用负对数部分似然损失；所有深度学习模型使用 Adam（lr=1×10⁻³、weight_decay=1×10⁻⁴），未使用早停，训练 epoch 数按模型固定为 30。",
    )

    add_heading(doc, "2.6 评估与统计检验", 2)
    add_body(
        doc,
        "所有模型使用 5 折交叉验证。分类任务报告 Accuracy 和 Macro-F1；生存任务报告 ROC AUC 和 C-index。对于 LogisticRegression/RandomForest/SVC/KNN/MLP 等模型，以预测概率或决策函数作为风险分数；对于 CoxPH，以其部分风险分数作为风险分数。为比较同一折上的模型，采用配对 t 检验和 Wilcoxon 符号秩检验。所有预处理、标准化、特征筛选和训练均在训练折内完成，测试折仅用于评估。",
    )
    add_body(doc, "Accuracy 为预测正确样本占全部样本的比例；Macro-F1 对每个类别分别计算 F1 后取算术平均，以降低类别不平衡的影响。ROC AUC 以死亡风险分数作为连续预测值，结合二分类事件标签计算。模型输出风险分数，数值越大表示死亡风险越高；计算 C-index 时使用该风险分数的负值，使 C-index 度量预测风险排序与真实生存顺序的一致性。")
    add_numbered_formula(doc, "C-index = P( R(x_i) > R(x_j) | T_i < T_j, δ_i = 1 )", 3)
    add_body(doc, "统计检验仅比较同一交叉验证折上的成对分数，以减少折间变异干扰。文中报告配对 t 检验和 Wilcoxon 符号秩检验 p 值；显著性阈值取 0.05。")
    add_body(doc, "分析使用 Python 3.11 及相关科学计算库完成：scikit-learn 用于传统机器学习与交叉验证，PyTorch 用于 MLP、FullSizeCNN 和深度融合模型，lifelines 用于 CoxPH 与 C-index，SHAP 和梯度 saliency 用于可解释性。固定随机种子为 42。")
    add_metrics_note(doc)

    add_heading(doc, "3. 结果", 1)
    add_body(doc, "正文展示代表性结果；完整逐模型、逐折均值和标准差见补充材料表 S4–S23。")
    add_heading(doc, "3.1 组学图像示例", 2)
    add_body(
        doc,
        "图 3 展示一个 PAM50 mRNA 样本和一个 Survival mRNA 样本的灰度表达图、NSRE 权重图和基因功能类别图。图像中心区域对应高 NSRE 特征，说明视觉上的中心区域直接编码了模型认为对结局判别更重要的基因。基因功能类别图以颜色标记基因功能类别，与 NSRE 灰度图位置严格对应，便于后续把卷积模型关注区域转化为生物学解释。",
    )
    add_figure(doc, FIG / "fig3_example_v8.png", "Figure 3. Omics image examples. A1-A3: one PAM50 mRNA sample shown as a 20×20 grayscale expression map, NSRE importance map, and gene functional category map. B1-B3: one survival mRNA sample shown as a 15×15 grayscale expression map, NSRE importance map, and gene functional category map. Axes indicate pixel positions; high-NSRE features are placed centrally, and the functional category map aligns pixel-wise with the expression and NSRE maps.", width=6.4)

    add_heading(doc, "3.2 单组学预测", 2)
    add_body(doc, "Table 3 summarizes the overall best model for each task on full samples; Figure 4 shows all compared models with mean±SD. Full single-omics results are in Supplementary Table S4, and intersection single-omics baselines in Supplementary Table S5.")
    add_body(
        doc,
        "单组学结果显示，mRNA 对 PAM50 的判别能力最强，miRNA 次之，CNV 最弱；生存预测中 mRNA 与 CNV 的 C-index 接近，miRNA 较弱。各任务、各指标下的最优模型及其 5 折交叉验证 mean±SD 见表 3，完整模型比较见图 4 与补充表 S4。",
    )
    single_all = read_tsv("comprehensive_batch1_single_omics_results.tsv")
    single_best_rows = []
    for task, metric in [("PAM50", "accuracy"), ("PAM50", "macro_f1"), ("Survival", "roc_auc"), ("Survival", "c_index")]:
        sub = single_all[(single_all["task"] == task) & (single_all["metric"] == metric)]
        sub = sub.sort_values("mean", ascending=False)
        best = sub.iloc[0]
        single_best_rows.append(
            {
                "Task": task,
                "Metric": metric,
                "Best model": f"{best['omics']} {best['model']}",
                "Mean": float(best["mean"]),
                "Std": float(best["std"]),
            }
        )
    single_best = pd.DataFrame(single_best_rows)
    add_dataframe_table(doc, single_best, [1.0, 1.1, 1.8, 0.8, 0.8])
    add_caption(doc, "Table 3. Overall best single-omics models on full samples (5-fold CV, mean±SD; features selected within each training fold). Per-omics and per-model detail is in Supplementary Table S4.")
    add_figure(doc, FIG / "fig3_single_omics_v3.png", "Figure 4. Single-omics prediction on full samples. A and B: PAM50 Accuracy and Macro-F1; C and D: survival ROC AUC and C-index. Bars show 5-fold cross-validation mean±SD with features selected within each training fold. PAM50 sample sizes: mRNA 833, CNV 818, miRNA 497; survival: mRNA 1073, CNV 1057, miRNA 739.", width=6.4)

    add_heading(doc, "3.3 多组学整合", 2)
    add_body(doc, "Figure 5 shows the triple-omics fusion methods; full fusion results are in Supplementary Table S6, and the best fusion versus the corresponding best single-omics comparison is summarized in Supplementary Table S10.")
    add_body(
        doc,
        "在 493 例三组学交集样本上，PAM50 的最优融合为 Transformer（Accuracy 0.9270±0.0132、Macro-F1 0.9288±0.0128）；简单拼接方法中 Concat SVC 的 Accuracy 与 Macro-F1 分别为 0.9168±0.0120、0.9125±0.0162。与交集最优单组学 mRNA LogisticRegression（0.9270±0.0217、0.9184±0.0316）相比，Transformer 在 Accuracy 上持平，Macro-F1 略优，但未形成统计显著优势。",
    )
    add_body(
        doc,
        "生存预测中，CNN 晚期平均融合的 ROC AUC 为 0.7591±0.0619，三组学 Concat MLP 的 C-index 为 0.7588±0.0492；两者分别高于交集最优单组学 mRNA MLP（ROC AUC 0.7258±0.0690）和 mRNA FullSizeCNN（C-index 0.7448±0.0705），但统计检验显示提升未达到显著水平（补充表 S10）。",
    )
    add_figure(doc, FIG / "fig4_triple_integration_v2.png", "Figure 5. Triple-omics fusion results. A and B: PAM50 Accuracy and Macro-F1; C and D: survival ROC AUC and C-index. Error bars represent 5-fold cross-validation SD. PAM50 intersection n=493; survival intersection n=731.", width=6.4)

    add_heading(doc, "3.4 两两组学、Stacking 与复杂方法", 2)
    add_body(
        doc,
        "两两组学中，PAM50 的 mRNA+CNV LogisticRegression 达到 Accuracy 0.9633±0.0039；mRNA+miRNA LogisticRegression 为 0.9095；CNV+miRNA LogisticRegression 为 0.8438。生存预测中，mRNA+CNV Concat MLP 的 C-index 为 0.7451，mRNA+miRNA Concat MLP 为 0.7337，CNV+miRNA CNN LateAvg 为 0.7216。9 模型 Stacking 的 PAM50 Accuracy 为 0.9229，生存 C-index 为 0.7321、ROC AUC 为 0.7469，未稳定超过最优简单融合。",
    )
    add_body(
        doc,
        "在复杂方法中，Transformer 在 PAM50 上表现较好（Accuracy 0.9270、Macro-F1 0.9288），但生存 C-index 为 0.6938；Multi-task、DeepSurv、低秩双线性和 GNN 未在当前小样本上显示出稳定提升。",
    )
    add_figure(doc, FIG / "fig5_advanced_methods_v2.png", "Figure 6. Advanced integration methods. A and B: PAM50 Accuracy and Macro-F1; C and D: survival ROC AUC and C-index. Methods include Multi-task, DeepSurv, Transformer, low-rank bilinear fusion, and GNN. Error bars represent 5-fold cross-validation SD.", width=6.4)

    add_heading(doc, "3.5 统计检验", 2)
    add_body(doc, "Supplementary Table S10 lists paired t-test and Wilcoxon results between the best fusion and the corresponding best single-omics model on the same folds; pairwise fusion results are in Supplementary Table S7, and stacking and complex methods in Supplementary Tables S8-S9.")
    add_body(
        doc,
        "对最优融合与对应最优单组学进行配对检验：PAM50 上 Transformer 与 mRNA LogisticRegression 的 Accuracy 差值为 0.0000（配对 t 检验 p=0.9979，Wilcoxon p=0.8750），Macro-F1 差值为 0.0104（p=0.6125，Wilcoxon p=1.0000）；生存上 CNN 晚期平均融合与 mRNA MLP 的 ROC AUC 差值为 0.0333（p=0.0930，Wilcoxon p=0.1250），三组学 Concat MLP 与 mRNA FullSizeCNN 的 C-index 差值为 0.0140（p=0.5480，Wilcoxon p=0.4375）。因此，所有最优融合均未较对应最优单组学达到统计显著优势。",
    )
    add_heading(doc, "3.6 可解释性与关键因子", 2)
    add_body(
        doc,
        "可解释分析中，SHAP 采用 scikit-learn LogisticRegression 的 LinearExplainer，以当前训练折全部样本作为背景（background）计算测试折样本的加性归因；CNN saliency 采用输入梯度在 5 折上的聚合；功能富集以 mRNA 全转录组基因作为背景基因集（universe），通过 g:Profiler 进行 GO/KEGG/Reactome 富集，多重检验采用 g:Profiler 默认校正。",
    )
    add_body(
        doc,
        "SHAP 分析显示，mRNA PAM50 LogisticRegression 的关键基因包括 CYP2B7P1、TFF1、C1orf64、AGR3、KCNJ3、MIA、PPP1R14C、ESR1 和 STAC2。CNN saliency 与 SHAP 的共识基因包括 TCAM1P、MIA、SMC1B、PPP1R14C、KLK6、SLC6A11、FABP7 和 FOXC1。生存单变量 Cox 分析识别出 MS4A1、COL17A1、C2orf40、PLA2G2D、FABP7、CCL19 和 GZMB。通路富集提示雌激素信号、细胞增殖调控、细胞分裂和上皮发育等生物学过程；SHAP 与 saliency 单图见补充材料图 S3–S7。",
    )
    add_figure(doc, FIG / "fig7_nsre_v3.png", "Figure 7. NSRE-based feature-mining visualization. A-C: grayscale expression, NSRE importance, and functional category maps; D-F: SHAP importance, CNN saliency, and functional-category/saliency overlay maps. Axes indicate pixel positions; the NSRE imaging scheme aligns expression, importance, function, and model attention in a common pixel coordinate system.", width=6.4)

    add_heading(doc, "3.7 功能类别遮盖与可解释性分析", 2)
    add_body(
        doc,
        "为进一步解释组学图像中不同功能区域对模型的贡献，本研究在方案2（功能大类分块 + 中心高重要性 + 中心向外螺旋填充）图像上实施功能类别遮盖实验。与最初的 NSRE 全局排序图像不同，方案2先按基因功能类别分块，再在块内按 NSRE 降序，并以块平均重要性由中心向外螺旋填充，因此同一颜色区域对应同一功能模块，中心区域对应高 NSRE 模块。该设计使图像像素邻接具有更明确的生物学含义，并支持对连续功能区域进行模块级遮挡。",
    )
    add_figure(doc, DATA / "interpretability/figures/fig_scheme2_masking_flow_en_v2.png", "Figure 8. Scheme-2 functional-category masking workflow. The target category pixels are set to zero, FullSizeCNN is retrained, and PAM50/survival performance is compared with the unmasked baseline. A, B, and C show the before-masking expression map, functional category map, and after-masking map, respectively.", width=6.4)
    add_body(
        doc,
        "对 mRNA、CNV 和 miRNA 分别遮盖 Development/Epithelium、Signaling/Transport、Hormone/Metabolism、Cell cycle/Proliferation 和 Other 等类别。mRNA PAM50 基线 Accuracy 为 0.9244±0.0124，Macro-F1 为 0.9069±0.0133；遮盖 Other 后 Macro-F1 降至 0.8819±0.0210。mRNA 生存基线 C-index 为 0.6984±0.0756，遮盖 Other 后降至 0.6757±0.0715。CNV 和 miRNA 同样显示 Other 类遮盖后性能下降最明显，但由于 Other 类特征数量较多，下降幅度同时受模块大小影响；mRNA 结果见补充材料表 S14，CNV/miRNA 结果见补充材料表 S15。",
    )
    add_figure(doc, DATA / "interpretability/figures/fig_other_omics_category_masking_v3.png", "Figure 9. CNV and miRNA functional-category masking results. A-D: CNV PAM50, CNV Survival, miRNA PAM50, and miRNA Survival Accuracy or C-index. Error bars represent 5-fold cross-validation SD.", width=6.4)
    add_body(
        doc,
        "为区分“功能模块特异性贡献”与“特征数量/区域大小效应”，对 mRNA 每个类别设置同数量随机基因遮盖对照。结果显示，功能类别遮盖与等量随机遮盖的性能差异在多数比较中未达到显著水平。对 mRNA PAM50 Other 基因重新富集发现，其仍显著富集于上皮细胞分化、上皮发育、组织发育、雌激素信号通路和细胞群体增殖等过程。SHAP 与 CNN saliency 共识基因的类别交叉分析显示，Top50 共识基因主要落在 Development/Epithelium 和 Other 中，其次是 Cell cycle/Proliferation、Signaling/Transport 和 Hormone/Metabolism。配对检验未显示大多数功能类别遮盖与基线之间存在稳健显著差异；折级指标见补充材料表 S16，配对检验见补充材料表 S17，Other 类二次富集见补充材料表 S18，共识基因重叠见补充材料表 S19。",
    )

    add_heading(doc, "4. 讨论", 1)
    add_body(
        doc,
        "在本数据集中，mRNA 是 PAM50 分型的主要信号源，其 LogisticRegression 和 SVC 已接近高精度，而 FullSizeCNN 在小样本下也达到 0.93 以上 Accuracy。CNV 单独预测能力有限，但在生存预测中 CoxPH 和 LogisticRegression 的 C-index 与 mRNA 接近，提示拷贝数变异可能携带与长期结局相关的补充信息。miRNA 的样本量较小，单组学性能较低。",
    )
    add_body(
        doc,
        "为区分“PAM50 分型预测”与“PAM50 基因面板重建”，本研究在 mRNA PAM50 任务上进行了 50 基因剔除敏感性分析：将最终 400 个 NSRE 排序特征中与经典 PAM50 50 基因重叠的 21 个基因剔除后，LogisticRegression 的 5 折 Accuracy 保持不变（0.9628±0.0138），MLP 与 FullSizeCNN 仅有轻微波动；而仅保留这 21 个 PAM50 基因时，LogisticRegression Accuracy 降至 0.9220±0.0076。该结果表明，NSRE 筛选出的 mRNA 特征集所携带的判别信息并非主要来自经典 PAM50 基因面板，而是覆盖了更广的转录组信号，因此本研究的 mRNA PAM50 任务应理解为对既有分子亚型的重建（subtype reconstruction），而非对 PAM50 50 基因面板的简单复现，也不宜被解释为独立的临床预后预测器。",
    )
    add_body(
        doc,
        "多组学融合对 PAM50 的增益有限，主要原因是 mRNA 已具有强判别能力，其他组学在高维小样本条件下容易引入噪声。生存任务中，简单拼接 MLP 和 CNN 晚期平均融合的均值优于单组学，说明多组学在生存风险预测中的互补性更强；但配对 t 检验与 Wilcoxon 检验均未达到 0.05 显著性水平（ROC AUC 的配对 t 检验 p=0.093），提示当前样本量不足以稳定区分融合策略。",
    )
    add_body(
        doc,
        "NSRE 图像化的优势在于提供可解释性，而不是仅仅追求最高预测精度。传统 CNN 将基因排列成任意二维结构，像素邻接关系可能缺乏生物学意义；本研究通过把 NSRE 高分基因放在中心，并同时绘制基因功能类别图和模型 saliency 图，使卷积关注区域可以直观映射到关键基因和通路。",
    )
    add_body(
        doc,
        "为进一步检验二维空间结构本身是否带来额外增益，本研究训练了与 FullSizeCNN 参数量一致的 Dense 等价基线（Flatten→Dense(32)→Dense(64)→输出，首层参数量与全尺寸卷积核完全相同）。结果显示，该 Dense 基线在 PAM50 上整体优于或持平 FullSizeCNN（mRNA Accuracy 0.9568±0.0184 对 0.9316±0.0071；CNV 0.7238 对 0.7164；miRNA 0.8370 对 0.8230），而生存预测中两者互有高低且差异未超过交叉验证标准差（mRNA C-index 0.6802 对 0.7125）。因此，本数据中的判别信息主要来自原始特征数值本身，而非人为构造的二维邻接结构；FullSizeCNN 图像化的价值在于把特征重要性、功能类别和模型注意力统一到同一像素坐标系，从而支撑可解释分析，而非提升预测精度。",
    )
    add_body(
        doc,
        "作为对排序假设的直接检验，本研究在 mRNA 上对 NSRE/JSD 螺旋排序与 20 次随机基因排列进行了对照。结果表明，NSRE 螺旋排序并未优于随机排列：PAM50 Accuracy 的 NSRE 结果为 0.9316，而 20 次随机排列的均值为 0.9267（范围 0.9112–0.9400，经验 p=0.25）；Survival C-index 的 NSRE 结果为 0.7125，随机排列均值为 0.7055（经验 p=0.35）。均值表达排序在 PAM50 Accuracy 上与 NSRE 完全一致（0.9316），进一步说明二维像素位置本身几乎不携带预测信息。据此，本文将组学图像化明确限定为一种“可解释的空间表示”，而不是能提升分类或生存预测精度的排序方法。",
    )
    add_body(
        doc,
        "面向 Molecular Stress Responses and Adaptive Reprogramming 这一目标主题，本研究在全转录组层面进一步考察了 Hypoxia、ROS、OXPHOS、UPR、mTORC1、Glycolysis、EMT 和 DNA repair 八条应激/适应性重编程通路与 JSD 高重要性基因、PAM50 亚型及生存风险的关系。结果显示，这些通路在 PAM50 任务的前 500 个 JSD 高重要性基因中均未显著富集（mTORC1 的 Fisher p=0.063 为边缘结果，其余 p≥0.32），单变量 Cox 分析也未发现任一通路评分与总生存显著相关（DNA repair HR 0.789、p=0.104 为最强但未达显著）。通路评分在 PAM50 亚型间高度差异（Kruskal-Wallis p<0.001），但这在表达定义亚型的背景下是预期现象。因此，在当前的 bulk 表达、精选标志基因框架下，尚未观察到这些经典应激通路对判别或预后的独立贡献；相关结果应作为探索性阴性发现，并提示后续需在单细胞、空间转录组或通路过表达/敲低层面进一步验证。乳腺癌转移中的翻译适应与应激重编程机制为后续研究提供了更广的生物学背景[49]。",
    )
    add_body(
        doc,
        "与已有“表格数据→图像”的表示方法相比，DeepInsight 通过 t-SNE 把特征点布局为二维图像以利用 CNN 的局部感受野，DeepInsight-3D 进一步将其推广到多组学三维表示，Yan 等则基于转录组特征图进行癌型与生存预测[32-34]；SurvConvMixer 利用通路级基因表达图像进行癌症生存预测[50]，MoACNN-XGNet 则构建面向乳腺癌亚型的可解释多组学卷积网络[51]。本研究与这些工作的关键区别在于：其一，本研究使用监督式 JSD/NSRE 而非无监督降维来指导空间布局，并把基因功能类别与模型注意力对齐到同一像素坐标系，从而将空间位置直接绑定到可解释的生物学注释；其二，本文的 Dense 等价与随机排列对照表明，在本数据规模下二维空间布局本身并不带来预测增益，因此图像化被定位为可解释表示而非性能提升手段。这一结论与 DeepInsight 系列“图像化有助于 CNN 建模”的动机形成互补而非冲突——在更大样本或真实空间先验（如通路网络）可用时，空间结构才更可能产生增益。",
    )
    add_body(
        doc,
        "功能类别遮盖实验表明，Other 类在 mRNA、CNV 和 miRNA 中均表现出最大的性能下降，但其特征数量通常最多；等量随机遮盖对照未显示功能模块遮盖稳定优于随机遮盖。因此，Other 类的较大下降更可能是高维未注释特征集中携带判别信号，同时叠加了模块大小效应，而不是单纯由某一已知功能类别驱动。CNV 和 miRNA 的功能类别为规则化注释，尤其 CNV 部分类别特征数较少，相关结果应作为方向性证据。",
    )
    add_body(
        doc,
        "局限包括：仅使用 TCGA-BRCA 单中心回顾性数据，缺少独立外部队列；生存事件数量较少，类别不平衡；螺旋位置仍然不是真实生物学空间；复杂模型可能过拟合。统计推断方面，配对检验仅基于 5 折（n=5）的折级分数，统计功效有限，且未进行多重比较校正，因此边缘显著结果（如生存 ROC AUC 的配对 t 检验 p=0.093）需要谨慎解释；后续应采用 repeated CV、bootstrap 置信区间或折级效应量（如 Cohen's d）来增强证据强度。此外，特征数量选择与最终评估共用同一交叉验证，存在选择偏倚，理想情况下应使用嵌套交叉验证或独立验证集。",
    )
    add_body(doc, "与已有研究相比，本工作的贡献不在于简单追求最高分类准确率，而在于提供一种可追溯、可映射到关键基因的组学图像化表示。该表示既保留原始特征值，又通过 NSRE 将判别重要性显式编码到图像空间，为后续关键因子挖掘和临床转化提供了统一接口。")

    add_heading(doc, "5. 结论", 1)
    add_body(
        doc,
        "本研究建立了 JSD/NSRE 引导的组学图像化与多组学融合框架，能够同时支持乳腺癌 PAM50 四分类和总生存预测。mRNA 单组学在分子分型重建（subtype reconstruction）中表现最优，且其判别信息并非主要依赖经典 PAM50 基因面板；生存预测更依赖多组学互补。FullSizeCNN 为组学图像化提供了轻量、可解释的基线。未来可在外部数据和更稳健融合策略下进一步验证。",
    )

    add_heading(doc, "参考文献与数据来源", 1)
    add_body(doc, "[1] Perou CM, Sørlie T, Eisen MB, van de Rijn M, Jeffrey SS, Rees CA, et al. Molecular portraits of human breast tumours. Nature. 2000;406(6797):747-752. doi:10.1038/35021093. PMID:10963602.")
    add_body(doc, "[2] Sørlie T, Perou CM, Tibshirani R, Aas T, Geisler S, Johnsen H, et al. Gene expression patterns of breast carcinomas distinguish tumor subclasses with clinical implications. Proc Natl Acad Sci U S A. 2001;98(19):10869-10874. doi:10.1073/pnas.191367098. PMID:11553815.")
    add_body(doc, "[3] Parker JS, Mullins M, Cheang MCU, Leung S, Voduc D, Vickery T, et al. Supervised risk predictor of breast cancer based on intrinsic subtypes. J Clin Oncol. 2009;27(8):1160-1167. doi:10.1200/JCO.2008.18.1370. PMID:19204204.")
    add_body(doc, "[4] Cancer Genome Atlas Network. Comprehensive molecular portraits of human breast tumours. Nature. 2012;490(7418):61-70. doi:10.1038/nature11412. PMID:23000897.")
    add_body(doc, "[5] Curtis C, Shah SP, Chin SF, Turashvili G, Rueda OM, Dunning MJ, et al. The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups. Nature. 2012;486(7403):346-352. doi:10.1038/nature10983. PMID:22522925.")
    add_body(doc, "[6] Lehmann BD, Bauer JA, Chen X, Sanders ME, Chakravarthy AB, Shyr Y, et al. Identification of human triple-negative breast cancer subtypes and preclinical models for selection of targeted therapies. J Clin Invest. 2011;121(7):2750-2767. doi:10.1172/JCI45014. PMID:21633166.")
    add_body(doc, "[7] Ciriello G, Gatza ML, Beck AH, Wilkerson MD, Rhie SK, Pastore A, et al. Comprehensive molecular portraits of invasive lobular breast cancer. Cell. 2015;163(2):506-519. doi:10.1016/j.cell.2015.09.033. PMID:26451490.")
    add_body(doc, "[8] Cancer Genome Atlas Research Network, Weinstein JN, Collisson EA, Mills GB, Shaw KRM, Ozenberger BA, et al. The Cancer Genome Atlas Pan-Cancer analysis project. Nat Genet. 2013;45(10):1113-1120. doi:10.1038/ng.2764. PMID:24071849.")
    add_body(doc, "[9] Hoadley KA, Yau C, Wolf DM, Cherniack AD, Tamborero D, Ng S, et al. Multiplatform analysis of 12 cancer types reveals molecular classification within and across tissues of origin. Cell. 2014;158(4):929-944. doi:10.1016/j.cell.2014.06.049. PMID:25109877.")
    add_body(doc, "[10] Subramanian I, Verma S, Kumar S, Jere A, Anamika K. Multi-omics data integration, interpretation, and its application. Bioinform Biol Insights. 2020;14:1177932219899051. doi:10.1177/1177932219899051. PMID:32076369.")
    add_body(doc, "[11] Cantini L, Zakeri P, Hernandez C, Naldi A, Thieffry D, Remy E, et al. Benchmarking joint multi-omics dimensionality reduction approaches for the study of cancer. Nat Commun. 2021;12(1):124. doi:10.1038/s41467-020-20430-7. PMID:33402734.")
    add_body(doc, "[12] Guyon I, Elisseeff A. An introduction to variable and feature selection. J Mach Learn Res. 2003;3:1157-1182.")
    add_body(doc, "[13] Tibshirani R. Regression shrinkage and selection via the lasso. J R Stat Soc Series B Stat Methodol. 1996;58(1):267-288. doi:10.1111/j.2517-6161.1996.tb02080.x.")
    add_body(doc, "[14] Zou H, Hastie T. Regularization and variable selection via the elastic net. J R Stat Soc Series B Stat Methodol. 2005;67(2):301-320. doi:10.1111/j.1467-9868.2005.00503.x.")
    add_body(doc, "[15] Peng H, Long F, Ding C. Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy. IEEE Trans Pattern Anal Mach Intell. 2005;27(8):1226-1238. doi:10.1109/TPAMI.2005.159. PMID:16119262.")
    add_body(doc, "[16] Battiti R. Using mutual information for selecting features in supervised neural net learning. IEEE Trans Neural Netw. 1994;5(4):537-550. doi:10.1109/72.298224. PMID:18267827.")
    add_body(doc, "[17] Kursa MB, Rudnicki WR. Feature selection with the Boruta package. J Stat Softw. 2010;36(11):1-13. doi:10.18637/jss.v036.i11.")
    add_body(doc, "[18] Breiman L. Random forests. Mach Learn. 2001;45(1):5-32. doi:10.1023/A:1010933404324.")
    add_body(doc, "[19] Cortes C, Vapnik V. Support-vector networks. Mach Learn. 1995;20(3):273-297. doi:10.1007/BF00994018.")
    add_body(doc, "[20] Friedman JH. Greedy function approximation: a gradient boosting machine. Ann Stat. 2001;29(5):1189-1232. doi:10.1214/aos/1013203451.")
    add_body(doc, "[21] Libbrecht MW, Noble WS. Machine learning applications in genetics and genomics. Nat Rev Genet. 2015;16(6):321-332. doi:10.1038/nrg3920. PMID:25948244.")
    add_body(doc, "[22] Ching T, Himmelstein DS, Beaulieu-Jones BK, Kalinin AA, Do BT, Way GP, et al. Opportunities and obstacles for deep learning in biology and medicine. J R Soc Interface. 2018;15(141):20170387. doi:10.1098/rsif.2017.0387. PMID:29618526.")
    add_body(doc, "[23] Zou J, Huss M, Abid A, Mohammadi P, Torkamani A, Telenti A. A primer on deep learning in genomics. Nat Genet. 2019;51(1):12-18. doi:10.1038/s41588-018-0295-5. PMID:30478442.")
    add_body(doc, "[24] Sui W, Wang J, Miao D, Jiang Y, Liu G, Yang S, et al. The Evolution of Modeling Approaches: From Statistical Models to Deep Learning for Locust and Grasshopper Forecasting. Insects. 2026;17(2):182. doi:10.3390/insects17020182.")
    add_body(doc, "[25] LeCun Y, Bottou L, Bengio Y, Haffner P. Gradient-based learning applied to document recognition. Proc IEEE. 1998;86(11):2278-2324. doi:10.1109/5.726791.")
    add_body(doc, "[26] Krizhevsky A, Sutskever I, Hinton GE. ImageNet classification with deep convolutional neural networks. In: Advances in Neural Information Processing Systems. 2012;25:1097-1105.")
    add_body(doc, "[27] He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016:770-778. doi:10.1109/CVPR.2016.90.")
    add_body(doc, "[28] Hu J, Shen L, Sun G. Squeeze-and-excitation networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018:7132-7141. doi:10.1109/CVPR.2018.00745.")
    add_body(doc, "[29] Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015. 2015:234-241. doi:10.1007/978-3-319-24574-4_28.")
    add_body(doc, "[30] Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, et al. Attention is all you need. In: Advances in Neural Information Processing Systems. 2017;30:5998-6008. arXiv:1706.03762.")
    add_body(doc, "[31] Shi F, Wang M, Teng Z, Cai L, Liu G, Xing Y, et al. HDGS-Net: nucleosome occupancy prediction based on a hybrid dilated gated separable convolutional neural network. BMC Genomics. 2026;27(1):209. doi:10.1186/s12864-026-12523-2.")
    add_body(doc, "[32] Sharma A, Vans E, Shigemizu D, Boroevich KA, Tsunoda T. DeepInsight: a methodology to transform a non-image data to an image for convolution neural network architecture. Sci Rep. 2019;9(1):11399. doi:10.1038/s41598-019-47765-6. PMID:31388036.")
    add_body(doc, "[33] Sharma A, Lysenko A, Boroevich KA, Tsunoda T. DeepInsight-3D architecture for anti-cancer drug response prediction with deep-learning on multi-omics. Sci Rep. 2023;13(1):2483. doi:10.1038/s41598-023-29644-3. PMID:36774402.")
    add_body(doc, "[34] Yan M, Dong Z, Zhu Z, Qiao C, Wang M, Teng Z, et al. Cancer type and survival prediction based on transcriptomic feature map. Comput Biol Med. 2025;192:110267. doi:10.1016/j.compbiomed.2025.110267. PMID:40311464.")
    add_body(doc, "[35] Cox DR. Regression models and life-tables. J R Stat Soc Series B Stat Methodol. 1972;34(2):187-202. doi:10.1111/j.2517-6161.1972.tb00899.x.")
    add_body(doc, "[36] Harrell FE Jr, Lee KL, Califf RM, Pryor DB, Rosati RA. Evaluating the yield of medical tests. JAMA. 1982;247(18):2543-2546. doi:10.1001/jama.1982.03320430047030. PMID:7069920.")
    add_body(doc, "[37] Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data. Stat Med. 2011;30(10):1105-1117. doi:10.1002/sim.4154. PMID:21484848.")
    add_body(doc, "[38] Ishwaran H, Kogalur UB, Blackstone EH, Lauer MS. Random survival forests. Ann Appl Stat. 2008;2(3):841-860. doi:10.1214/08-AOAS169.")
    add_body(doc, "[39] Katzman JL, Shaham U, Cloninger A, Bates J, Jiang T, Kluger Y. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC Med Res Methodol. 2018;18(1):24. doi:10.1186/s12874-018-0482-1. PMID:29482517.")
    add_body(doc, "[40] Mitchel J, Chatlin K, Tong L, Wang MD. A translational pipeline for overall survival prediction of breast cancer patients by decision-level integration of multi-omics data. In: Proceedings IEEE International Conference on Bioinformatics and Biomedicine. 2019:1573-1580. doi:10.1109/BIBM47256.2019.8983243. PMID:32601549.")
    add_body(doc, "[41] Goodfellow I, Pouget-Abadie J, Mirza M, Xu B, Warde-Farley D, Ozair S, et al. Generative adversarial nets. In: Advances in Neural Information Processing Systems. 2014;27:2672-2680. arXiv:1406.2661.")
    add_body(doc, "[42] Arjovsky M, Chintala S, Bottou L. Wasserstein generative adversarial networks. In: Proceedings of the 34th International Conference on Machine Learning. PMLR. 2017;70:214-223.")
    add_body(doc, "[43] Gulrajani I, Ahmed F, Arjovsky M, Dumoulin V, Courville A. Improved training of Wasserstein GANs. In: Advances in Neural Information Processing Systems. 2017:5767-5777. arXiv:1704.00028.")
    add_body(doc, "[44] Lin J. Divergence measures based on the Shannon entropy. IEEE Trans Inf Theory. 1991;37(1):145-151. doi:10.1109/18.61115.")
    add_body(doc, "[45] Simonyan K, Vedaldi A, Zisserman A. Deep inside convolutional networks: visualising image classification models and saliency maps. arXiv:1312.6034. 2014.")
    add_body(doc, "[46] Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE International Conference on Computer Vision. 2017:618-626. doi:10.1109/ICCV.2017.74.")
    add_body(doc, "[47] Ribeiro MT, Singh S, Guestrin C. \"Why should I trust you?\": explaining the predictions of any classifier. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016:1135-1144. doi:10.1145/2939672.2939778.")
    add_body(doc, "[48] Lundberg SM, Lee SI. A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems. 2017;30:4765-4774. arXiv:1705.07874.")
    add_body(doc, "[49] Chen S, Navickas A, Goodarzi H. Translational adaptation in breast cancer metastasis and emerging therapeutic opportunities. Trends Pharmacol Sci. 2024;45(4):304-318. doi:10.1016/j.tips.2024.02.002.")
    add_body(doc, "[50] Wang S, Liu Y, Zhang H, Liu Z. SurvConvMixer: robust and interpretable cancer survival prediction based on ConvMixer using pathway-level gene expression images. BMC Bioinformatics. 2024;25(1):133. doi:10.1186/s12859-024-05745-2.")
    add_body(doc, "[51] Li Q, Liu L, Zhang Q, Zhang X, Li N, Zhao Y, et al. MoACNN-XGNet: interpretable multi-omics convolutional network for breast cancer subtyping and prognostic genes identification. IEEE J Biomed Health Inform. 2025. doi:10.1109/JBHI.2025.3595381.")

    add_body(doc, "Data availability statement：本研究使用的原始公开数据来自 The Cancer Genome Atlas（TCGA-BRCA），可通过 NCI Genomic Data Commons（GDC）数据门户获取（project TCGA-BRCA），并可经 UCSC Xena 平台访问（dataset: TCGA Breast Cancer (BRCA)）。")
    add_body(doc, "本研究的处理数据、特征选择与建模代码、NSRE/JSD 图像、结果表、补充材料及可复现说明已整理为代码与数据仓库并公开托管于 GitHub（repository: NSRE_omics_imaging_breast_cancer），仓库链接见正文末尾的 Code availability 条目。")
    add_body(doc, "Code availability：https://github.com/381521602/NSRE_omics_imaging_breast_cancer")
    add_body(doc, "本文为中文初稿；正式投稿前将按目标期刊格式补充编号引用、作者单位、基金和利益冲突声明。")

    doc.save(OUT_MAIN)
    print(OUT_MAIN)
    return doc


def export_supplementary_tables():
    """Export all data/*.tsv tables to CSV and package them with a manifest.

    This keeps the Word supplement as a narrative/visual summary while providing
    the complete machine-readable results as an external ZIP archive.
    """
    out_dir = DATA / "supplementary_data"
    csv_dir = out_dir / "tables"
    csv_dir.mkdir(parents=True, exist_ok=True)

    tsv_paths = sorted(
        p for p in DATA.rglob("*.tsv") if p.is_file() and out_dir not in p.parents
    )
    manifest_rows = []
    for src in tsv_paths:
        rel = src.relative_to(DATA).as_posix()
        csv_name = rel.replace("/", "__").replace("\\", "__").rsplit(".", 1)[0] + ".csv"
        try:
            df = pd.read_csv(src, sep="\t", low_memory=False)
        except Exception as exc:
            manifest_rows.append(
                {
                    "source_tsv": rel,
                    "csv_name": "",
                    "n_rows": "",
                    "n_cols": "",
                    "status": f"SKIPPED: {exc}",
                }
            )
            continue
        dst = csv_dir / csv_name
        df.to_csv(dst, index=False, encoding="utf-8-sig")
        manifest_rows.append(
            {
                "source_tsv": rel,
                "csv_name": f"tables/{csv_name}",
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "status": "OK",
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.tsv", sep="\t", index=False)
    archive = out_dir / "supplementary_data_tables.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_dir / "manifest.tsv", arcname="manifest.tsv")
        for csv_path in sorted(csv_dir.glob("*.csv")):
            zf.write(csv_path, arcname=f"tables/{csv_path.name}")
    print(archive)
    return archive


def build_supplement():
    doc = setup_document()
    add_paragraph(
        doc,
        "Supplementary Material: Data, Feature Selection, Single-Omics and Multi-Omics Modeling Results",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        size=15,
        bold=True,
        color=INK,
        after=4,
    )
    add_paragraph(
        doc,
        "补充材料：数据、特征筛选、单组学与多组学建模完整结果",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        size=12,
        bold=True,
        color=DARK,
        after=10,
    )

    add_heading(doc, "S1. 数据来源与样本清单", 1)
    add_body(
        doc,
        "数据来自 TCGA-BRCA 的公开平台 GDC 与 UCSC Xena[S1,S2]。mRNA 采用 HiSeqV2 基因表达矩阵；CNV 采用 GISTIC2 基因级阈值化拷贝数[S3]；miRNA 采用 miRNA 表达矩阵。样本优先选择原发肿瘤，按 TCGA barcode 对齐。",
    )
    add_dataframe_table(
        doc,
        pd.DataFrame(
            {
                "Task": ["PAM50", "Survival"],
                "mRNA": ["833", "1073"],
                "CNV": ["818", "1057"],
                "miRNA": ["497", "739"],
                "Triple intersection": ["493", "731"],
                "mRNA∩CNV": ["818", "1055"],
                "mRNA∩miRNA": ["497", "737"],
                "CNV∩miRNA": ["493", "733"],
            }
        ),
        [1.1, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0],
    )
    add_caption(doc, "Table S1. Sample sizes by omics and intersections")
    add_body(
        doc,
        "表 S1 仅给出样本数量。完整的样本级 provenance（case ID、sample type、mRNA/CNV/miRNA sample ID、PAM50 亚型、vital status、OS time/event 及各组学可用性）见 data/sample_level_provenance.tsv（共 1,098 例）。",
    )

    add_heading(doc, "S2. 特征提取算法", 1)
    add_body(
        doc,
        "本研究比较了 4 种基础特征筛选方法及其两两组合[S4]。低方差过滤（V）移除近常数特征；F 值（ANOVA F）衡量连续特征与分类标签的线性判别能力；L1/LASSO 通过线性 SVM 的稀疏系数进行嵌入式选择[S5]；JSD/NSRE 基于类别间分箱直方图的 Jensen–Shannon divergence。",
    )
    add_body(
        doc,
        "JSD 公式：D_JSD(P||Q)=0.5 Σ_x [P(x) log2(2P(x)/(P(x)+Q(x))) + Q(x) log2(2Q(x)/(P(x)+Q(x)))]。实际计算时 P、Q 为加 ε 后归一化的类别直方图；二分类直接计算两类间 JSD。PAM50 mRNA 和生存数据使用平均 pairwise JSD，PAM50 CNV 使用 one-vs-rest max JSD，PAM50 miRNA 先使用 pairwise JSD 粗筛、再以 L1 精筛。",
    )
    add_body(
        doc,
        "组合方式包括 V、V_F、V_L1、V_NSRE、V_F_L1、V_F_NSRE、V_L1_F、V_L1_NSRE、V_NSRE_F、V_NSRE_L1，其中 V 表示低方差预过滤，下划线后的顺序表示先粗筛后精筛或先粗筛后嵌入选择。",
    )
    add_dataframe_table(
        doc,
        pd.DataFrame(
            {
                "Method": [
                    "V",
                    "V_F",
                    "V_L1",
                    "V_NSRE",
                    "V_F_L1",
                    "V_F_NSRE",
                    "V_L1_F",
                    "V_L1_NSRE",
                    "V_NSRE_F",
                    "V_NSRE_L1",
                ],
                "Description": [
                    "Low-variance filter",
                    "Low-variance + ANOVA F",
                    "Low-variance + L1/LASSO",
                    "Low-variance + NSRE",
                    "Low-variance + F coarse + L1 fine",
                    "Low-variance + F coarse + NSRE fine",
                    "Low-variance + L1 coarse + F fine",
                    "Low-variance + L1 coarse + NSRE fine",
                    "Low-variance + NSRE coarse + F fine",
                    "Low-variance + NSRE coarse + L1 fine",
                ],
            }
        ),
        [1.4, 4.4],
    )
    add_caption(doc, "Table S2. Feature-selection algorithm combinations")

    add_heading(doc, "S3. 最终特征数量与数据集", 1)
    add_body(
        doc,
        "PAM50 最终特征数：mRNA 400、CNV 50、miRNA 600。生存最终特征数：mRNA 200、CNV 150、miRNA 600。特征名全部保留，数据文件见 data/final_datasets/。",
    )
    add_dataframe_table(
        doc,
        pd.DataFrame(
            {
                "Task": ["PAM50", "PAM50", "PAM50", "Survival", "Survival", "Survival"],
                "Omics": ["mRNA", "CNV", "miRNA", "mRNA", "CNV", "miRNA"],
                "Final features": ["400", "50", "600", "200", "150", "600"],
                "Image size": ["20×20", "8×8", "25×25", "15×15", "13×13", "25×25"],
                "Grid capacity": ["400", "64", "625", "225", "169", "625"],
                "Zero-padded": ["0", "14", "25", "25", "19", "25"],
            }
        ),
        [1.0, 0.8, 0.9, 1.1, 1.0, 0.8],
    )
    add_caption(doc, "Table S3. Final feature counts and NSRE image sizes")

    add_heading(doc, "S4. 单组学全部模型结果", 1)
    add_body(doc, "Table S4 reports full-sample single-omics 5-fold CV results based on final features selected within each training fold.")
    single = read_tsv("comprehensive_batch1_single_omics_results.tsv")
    add_dataframe_table(doc, single, [1.0, 1.1, 1.8, 1.3, 0.8, 0.8])
    add_caption(doc, "Table S4. Single-omics full-sample 5-fold CV results (mean±SD; features selected within each training fold)")

    add_heading(doc, "S5. 交集单组学基线", 1)
    add_body(doc, "为与多组学融合公平比较，在统一交集样本上重新计算单组学基线。")
    inter = read_tsv("comprehensive_intersection_single_omics_results.tsv")
    add_dataframe_table(doc, inter, [1.0, 1.1, 1.8, 1.3, 0.8, 0.8])
    add_caption(doc, "Table S5. Single-omics baselines on triple-omics intersection samples (mean±SD; features selected within each training fold)")

    add_heading(doc, "S6. 三组学融合结果", 1)
    triple = read_tsv("comprehensive_triple_integration_results.tsv")
    add_dataframe_table(doc, triple, [2.0, 1.2, 1.3, 0.8, 0.8])
    add_caption(doc, "Table S6. Triple-omics fusion 5-fold CV results (mean±SD; features selected within each training fold)")

    add_heading(doc, "S7. 两两组学融合结果", 1)
    pair = read_tsv("comprehensive_pairwise_integration_results.tsv")
    add_dataframe_table(doc, pair, [1.3, 2.1, 1.2, 1.3, 0.8, 0.8])
    add_caption(doc, "Table S7. Pairwise-omics fusion 5-fold CV results (mean±SD; features selected within each training fold)")

    add_heading(doc, "S8. 9 模型 Stacking 与复杂方法", 1)
    stacking = read_tsv("final_9model_stacking_results.tsv")
    add_dataframe_table(doc, stacking, [1.4, 1.2, 1.3, 0.8, 0.8])
    add_caption(doc, "Table S8. Nine-model Stacking and base-model results (mean±SD; features selected within each training fold)")

    adv_rows = []
    for method, fname in [
        ("Multi-task", "advanced_method1_multitask_results.tsv"),
        ("DeepSurv", "advanced_method2_deepsurv_results.tsv"),
        ("Transformer", "advanced_method3_transformer_results.tsv"),
        ("Low-rank bilinear", "advanced_method4_lowrank_bilinear_results.tsv"),
        ("GNN", "advanced_method5_gnn_results.tsv"),
    ]:
        df = read_tsv(fname)
        if "task" not in df.columns:
            df["task"] = "Survival"
        df["method"] = method
        adv_rows.append(df)
    adv = pd.concat(adv_rows, ignore_index=True)[["method", "task", "metric", "mean", "std"]]
    add_dataframe_table(doc, adv, [1.6, 1.2, 1.3, 0.8, 0.8])
    add_caption(doc, "Table S9. Complex integration-method results (mean±SD; features selected within each training fold)")

    add_heading(doc, "S9. 配对检验", 1)
    paired = read_tsv("comprehensive_main_best_paired_tests.tsv")
    fusion_disp_s = {
        ("PAM50", "accuracy"): "Transformer",
        ("PAM50", "macro_f1"): "Transformer",
        ("Survival", "roc_auc"): "CNN LateAvg",
        ("Survival", "c_index"): "Concat MLP",
    }
    single_disp_s = {
        ("PAM50", "accuracy"): "mRNA LogisticRegression",
        ("PAM50", "macro_f1"): "mRNA LogisticRegression",
        ("Survival", "roc_auc"): "mRNA MLP",
        ("Survival", "c_index"): "mRNA FullSizeCNN",
    }
    paired_s = paired.copy()
    paired_s["Best fusion"] = [fusion_disp_s[(r["task"], r["metric"])] for _, r in paired.iterrows()]
    paired_s["Best single-omics"] = [single_disp_s[(r["task"], r["metric"])] for _, r in paired.iterrows()]
    paired_s = paired_s[
        [
            "task",
            "metric",
            "Best fusion",
            "fusion_mean",
            "fusion_std",
            "Best single-omics",
            "single_mean",
            "single_std",
            "mean_diff",
            "t_pvalue",
            "wilcoxon_pvalue",
        ]
    ].copy()
    paired_s.columns = [
        "Task",
        "Metric",
        "Best fusion",
        "Fusion mean",
        "Fusion std",
        "Best single-omics",
        "Single mean",
        "Single std",
        "Fusion−single difference",
        "Paired t p",
        "Wilcoxon p",
    ]
    add_dataframe_table(doc, paired_s, [0.9, 1.0, 1.2, 0.8, 0.8, 1.3, 0.8, 0.8, 1.0, 0.8, 0.8])
    add_caption(doc, "Table S10. Paired tests between best fusion and best single-omics (5-fold CV, mean±SD; features selected within each training fold)")

    add_heading(doc, "S10. 图像化消融与结构对比", 1)
    add_body(doc, "不同排序方案的 FullSizeCNN 结果、经典全尺寸卷积结构和 WGAN-GP 数据增强结果如下。WGAN-GP 数据增强基于 GAN、Wasserstein GAN 和梯度惩罚训练策略实现[S6-S8]。")
    reorder = read_tsv("reorder_fullsize_cnn_results.tsv")
    add_dataframe_table(doc, reorder, [2.0, 1.2, 1.3, 0.8, 0.8])
    add_caption(doc, "Table S11. NSRE spiral ordering vs functional-reordering schemes (PAM50; mean±SD; features selected within each training fold)")

    classic = read_tsv("nsre_classic_structures_results.tsv")
    add_dataframe_table(doc, classic, [2.0, 1.2, 1.3, 0.8, 0.8])
    add_caption(doc, "Table S12. Classical full-size convolutional structures (mean±SD; features selected within each training fold)")

    gan = read_tsv("wgan_gp_augmentation_all_omics_results.tsv")
    add_dataframe_table(doc, gan, [1.0, 1.1, 1.1, 1.0, 0.8, 0.8])
    add_caption(doc, "Table S13. WGAN-GP augmentation (baseline vs 1x/5x/10x) with FullSizeCNN across mRNA, CNV, and miRNA (5-fold CV, mean±SD; features selected within each training fold)")
    add_figure(doc, DATA / "images/report_figures/fig_classic_structures.png", "Figure S1. Classical full-size convolutional structures. Accuracy/Macro-F1 and ROC AUC/C-index are compared for FullSizeCNN, GroupNorm+Mish, FCN, ResNet, and multi-branch SE; error bars show 5-fold CV SD.")
    add_figure(doc, DATA / "images/report_figures/fig_augmentation.png", "Figure S2. WGAN-GP augmentation for single-omics prediction. Comparisons include no augmentation, 1x, 5x, and 10x for mRNA, CNV, and miRNA; error bars show 5-fold CV SD.")

    add_body(doc, "为检验二维空间结构是否带来额外增益，训练了与 FullSizeCNN 首层参数量一致的 Dense 等价基线（Flatten→Dense(32)→Dense(64)→输出）。Dense 等价基线与 FullSizeCNN 的逐组学、逐任务、逐指标比较见 Table S14 与 Figure S3。")
    dense = read_tsv("dense_equivalent_baseline_results.tsv")
    cnn_full = read_tsv("comprehensive_batch1_single_omics_results.tsv")
    cnn_full = cnn_full[cnn_full["model"] == "FullSizeCNN"]
    dense_rows = []
    for omics in ["mRNA", "CNV", "miRNA"]:
        for task, metric in [("PAM50", "accuracy"), ("PAM50", "macro_f1"), ("Survival", "roc_auc"), ("Survival", "c_index")]:
            d = dense[(dense["omics"] == omics) & (dense["task"] == task) & (dense["metric"] == metric)].iloc[0]
            c = cnn_full[(cnn_full["omics"] == omics) & (cnn_full["task"] == task) & (cnn_full["metric"] == metric)].iloc[0]
            dense_rows.append(
                {
                    "Omics": omics,
                    "Task": task,
                    "Metric": metric,
                    "Dense-equivalent mean±SD": f"{d['mean']:.4f}±{d['std']:.4f}",
                    "FullSizeCNN mean±SD": f"{c['mean']:.4f}±{c['std']:.4f}",
                }
            )
    add_dataframe_table(doc, pd.DataFrame(dense_rows), [1.0, 1.1, 1.0, 1.6, 1.6])
    add_caption(doc, "Table S14. Dense-equivalent baseline vs FullSizeCNN (5-fold CV, mean±SD; features selected within each training fold)")
    add_figure(doc, FIG / "fig_dense_equivalent.png", "Figure S3. Dense-equivalent baseline vs FullSizeCNN. A: PAM50 Accuracy; B: Survival C-index. The dense network with an identical first-layer parameter count performs comparably to, or better than, FullSizeCNN, indicating that the 2-D spatial arrangement does not itself provide a consistent predictive gain; error bars show 5-fold CV SD.")

    add_body(doc, "为检验 NSRE/JSD 螺旋排序是否优于随机排序，对 mRNA 生成 20 次随机基因排列，并加入均值表达排序作为确定性对照。随机排列的分布、均值表达排序与 NSRE 螺旋排序的对比见 Table S15 与 Figure S4；经验 p 值为 20 次随机排列中不低于 NSRE 结果的占比。")
    rp = read_tsv("random_permutation_control_stats.tsv")
    rp.columns = ["Task", "Metric", "NSRE spiral", "Mean expression", "Random mean", "Random std", "Random min", "Random max", "p(random ≥ NSRE)"]
    add_dataframe_table(doc, rp, [0.9, 1.0, 0.9, 1.0, 0.9, 0.8, 0.8, 0.8, 1.0])
    add_caption(doc, "Table S15. Random-permutation ordering control for mRNA (5-fold CV; NSRE spiral vs 20 random orders vs mean-expression ordering)")
    add_figure(doc, FIG / "fig_random_permutation.png", "Figure S4. Random-permutation ordering control. A: PAM50 Accuracy; B: PAM50 Macro-F1; C: Survival ROC AUC; D: Survival C-index. Histograms show the distribution over 20 random gene orders; the NSRE spiral and mean-expression orderings fall within the null, so the spatial ordering does not itself improve prediction.")

    add_heading(doc, "S11. 可解释性补充图", 1)
    add_body(doc, "SHAP、梯度 saliency 与 Grad-CAM 分别用于全局特征归因、单样本像素/基因重要性解释和卷积梯度定位[S9-S11]。")
    add_figure(doc, DATA / "interpretability/figures/fig_shap_top.png", "Figure S5. Top-20 SHAP genes for mRNA PAM50 LogisticRegression. The x-axis is mean absolute SHAP contribution; the y-axis is gene name.")
    add_figure(doc, DATA / "interpretability/figures/fig_cnn_top.png", "Figure S6. Top-20 CNN saliency genes for mRNA PAM50. Gradient-based saliency ranks input pixels/genes and highlights the most attended image regions.")
    add_figure(doc, DATA / "interpretability/figures/fig_consensus_top.png", "Figure S7. Top-20 consensus genes between SHAP and CNN saliency.")
    add_figure(doc, DATA / "interpretability/figures/fig_survival_hr.png", "Figure S8. Top univariate Cox hazard ratios for survival genes. HR<1 indicates association with better survival; HR>1 indicates worse survival.")
    add_figure(doc, DATA / "interpretability/figures/fig_enrichment.png", "Figure S9. Pathway enrichment of consensus genes.")

    add_heading(doc, "S12. 方案2功能类别遮盖实验", 1)
    add_body(doc, "为解释功能模块对模型的贡献，在方案2图像上逐一遮盖各功能类别，比较 FullSizeCNN 的 PAM50 分类与生存预测性能。遮盖方式为将目标类别像素置零，其余类别保持不变。")
    scheme2_mask = read_tsv("scheme2_category_masking_results.tsv")
    add_dataframe_table(doc, scheme2_mask, [1.2, 1.6, 0.9, 1.0, 0.7, 0.7])
    add_caption(doc, "Table S16. mRNA scheme-2 functional-category masking results (mean±SD; features selected within each training fold)")
    add_figure(doc, DATA / "interpretability/figures/fig_scheme2_masking_flow_en_v2.png", "Figure S10. Scheme-2 functional-category masking workflow. A: before masking; B: functional category map; C: after masking Other.")
    add_figure(doc, DATA / "interpretability/figures/fig_scheme2_category_masking.png", "Figure S11. mRNA scheme-2 functional-category masking performance. A: PAM50 Accuracy; B: Survival C-index; error bars show 5-fold CV SD.")

    add_heading(doc, "S13. CNV 与 miRNA 功能类别遮盖实验", 1)
    add_body(doc, "需要说明，CNV 与 miRNA 的功能分类均为启发式注释（heuristic annotation），而非互斥的生物学本体。CNV 通过基因符号的关键词规则归入六大类别；miRNA 先由 MIMAT 映射至 hsa-miR 名称，再按其家族/已知功能归入同一六大类别[S12]。该规则仅覆盖部分基因，大量特征被归入 Other，因此相关遮盖结果应视为方向性证据而非确定的生物学结论。完整的特征→类别注释映射见 data/omics_functional_category_annotation.tsv。两种组学均采用与 mRNA 相同的遮盖-重训练-评估流程。")
    other_mask = read_tsv("other_omics_category_masking_results.tsv")
    add_dataframe_table(doc, other_mask, [0.9, 0.9, 1.4, 1.6, 0.8, 0.7, 0.7])
    add_caption(doc, "Table S17. CNV and miRNA functional-category masking results (mean±SD; features selected within each training fold)")
    add_figure(doc, DATA / "interpretability/figures/fig_other_omics_category_masking_v3.png", "Figure S12. CNV and miRNA functional-category masking performance. A-D: CNV PAM50, CNV Survival, miRNA PAM50, and miRNA Survival.")

    add_heading(doc, "S14. 等量随机遮盖对照与统计检验", 1)
    add_body(doc, "为控制特征数量和区域大小效应，对 mRNA 每个功能类别生成同数量随机基因遮盖对照。折级指标见补充文件，配对 t 检验与 Wilcoxon 检验结果见下表。")
    equal_folds = read_tsv("mrna_equal_mask_control_folds.tsv")
    add_dataframe_table(doc, equal_folds, [0.8, 0.6, 1.6, 1.0, 0.8])
    add_caption(doc, "Table S18. Fold-level metrics for mRNA equal-size random masking controls")
    equal_test = read_tsv("mrna_equal_mask_control_paired_tests.tsv")
    add_dataframe_table(doc, equal_test, [0.8, 1.4, 1.0, 0.9, 0.8, 0.9, 0.9])
    add_caption(doc, "Table S19. Paired tests between mRNA functional-category masks and baseline")
    add_figure(doc, DATA / "interpretability/figures/fig_mrna_equal_mask_control.png", "Figure S13. mRNA functional-category masks vs equal-size random masks.")

    add_heading(doc, "S15. Other 类二次富集与共识基因交叉验证", 1)
    add_body(doc, "对 mRNA PAM50 的 Other 类基因重新进行 GO/KEGG/Reactome 富集，并与 SHAP/CNN saliency Top50 共识基因进行功能类别交叉验证[S13-S15]。")
    other_enrich = read_tsv("interpretability/mrna_pam50_other_reenrichment.tsv")
    if not other_enrich.empty:
        other_enrich_top = other_enrich.head(12)[["source", "term_id", "term_name", "p_value", "intersection_size"]].copy()
        add_dataframe_table(doc, other_enrich_top, [0.7, 1.1, 2.6, 1.0, 1.0])
        add_caption(doc, "Table S20. Secondary functional enrichment of mRNA PAM50 Other genes (top 12)")
    overlap = read_tsv("interpretability/mrna_pam50_category_consensus_overlap.tsv")
    add_dataframe_table(doc, overlap, [1.4, 1.0, 1.1, 3.0])
    add_caption(doc, "Table S21. Overlap between SHAP/CNN saliency Top50 consensus genes and mRNA functional categories")

    add_heading(doc, "S16. 应激与适应性重编程通路分析", 1)
    add_body(
        doc,
        "面向 Molecular Stress Responses and Adaptive Reprogramming 主题，在全转录组（TCGA-BRCA HiSeqV2，20,530 基因）上对 Hypoxia、ROS、OXPHOS、UPR、mTORC1、Glycolysis、EMT、DNA repair 八条通路进行了探索性分析：以精选标志基因集合计算每样本通路评分，并用 Fisher 精确检验考察其在前 500 个 JSD 高重要性基因中的富集，用 Kruskal-Wallis 检验考察其与 PAM50 亚型的关联，用单变量 Cox 回归考察其与总生存的关联。",
    )
    sp = read_tsv("stress_pathway_analysis_results.tsv")
    sp.columns = [
        "Pathway", "Genes in transcriptome", "Genes in top-500", "Enrichment OR",
        "Fisher p", "PAM50 Kruskal-Wallis p", "Survival HR", "HR 95% lower",
        "HR 95% upper", "Cox p",
    ]
    add_dataframe_table(doc, sp, [1.0, 1.1, 0.9, 0.8, 0.8, 1.1, 0.7, 0.8, 0.8, 0.7])
    add_caption(doc, "Table S22. Stress / adaptive-reprogramming pathway analysis over the full mRNA transcriptome (Fisher enrichment in top-500 JSD genes; Kruskal-Wallis for PAM50; univariate Cox for overall survival)")
    add_figure(doc, FIG / "fig_stress_pathway.png", "Figure S14. Stress / adaptive-reprogramming pathway analysis. A: Fisher enrichment (-log10 p) of eight pathways among the top-500 JSD genes; B: univariate Cox hazard ratios for overall survival. No pathway is significantly enriched or prognostic; error bars show 95% confidence intervals.")

    add_heading(doc, "S17. PAM50 50 基因剔除敏感性分析", 1)
    add_body(
        doc,
        "为验证 mRNA PAM50 任务的判别信号是否主要来自经典 PAM50 50 基因面板，本研究在 mRNA PAM50 的最终 400 个 NSRE 排序特征上进行了剔除敏感性分析。三种配置分别为：Full 400 genes（完整 400 特征）、Exclude PAM50 genes（从 400 特征中删除与 PAM50 50 基因重叠的 21 个基因，剩 379 特征）和 PAM50 genes only（仅保留重叠的 21 个基因）。对每种配置重新计算 NSRE 排序并生成对应尺寸的灰度图，采用 LogisticRegression、MLP 和 FullSizeCNN 在统一 5 折分层交叉验证下评估 Accuracy 与 Macro-F1。",
    )
    gex = read_tsv("pam50_gene_exclusion_results.tsv")
    config_labels = {
        "full_transcriptome": "Full 400 genes",
        "exclude_pam50_genes": "Exclude PAM50 genes (379 genes)",
        "pam50_only": "PAM50 genes only (21 genes)",
    }
    gex_rows = []
    for cfg in ["full_transcriptome", "exclude_pam50_genes", "pam50_only"]:
        for model in ["LogisticRegression", "MLP", "FullSizeCNN"]:
            acc = gex[(gex["config"] == cfg) & (gex["model"] == model) & (gex["metric"] == "accuracy")]
            f1 = gex[(gex["config"] == cfg) & (gex["model"] == model) & (gex["metric"] == "macro_f1")]
            gex_rows.append(
                {
                    "Configuration": config_labels[cfg],
                    "Model": model,
                    "Accuracy": f"{acc['mean'].iloc[0]:.4f}±{acc['std'].iloc[0]:.4f}",
                    "Macro-F1": f"{f1['mean'].iloc[0]:.4f}±{f1['std'].iloc[0]:.4f}",
                }
            )
    add_dataframe_table(doc, pd.DataFrame(gex_rows), [2.0, 1.4, 1.4, 1.4])
    add_caption(doc, "Table S23. PAM50 50-gene exclusion sensitivity analysis for mRNA PAM50 (5-fold CV, mean±SD; features selected within each training fold)")
    add_figure(doc, FIG / "fig_pam50_gene_exclusion.png", "Figure S15. PAM50 50-gene exclusion sensitivity analysis. A: Accuracy; B: Macro-F1. Removing the 21 PAM50-overlapping genes barely changes performance, whereas the 21 PAM50 genes alone underperform the full 400-gene signature; error bars show 5-fold CV SD.")

    add_heading(doc, "S18. 数据与脚本文件", 1)
    for item in [
        "data/final_datasets/PAM50/{mRNA,CNV,miRNA}_PAM50_final.tsv",
        "data/final_datasets/Survival/{mRNA,CNV,miRNA}_Survival_final.tsv",
        "data/images/PAM50/{mRNA,CNV,miRNA}/images.npy 及 order.tsv",
        "data/images/Survival/{mRNA,CNV,miRNA}/images.npy 及 order.tsv",
        "data/pam50_gene_exclusion_results.tsv",
        "data/dense_equivalent_baseline_results.tsv",
        "data/random_permutation_control_results.tsv",
        "data/random_permutation_control_stats.tsv",
        "data/stress_pathway_analysis_results.tsv",
        "data/omics_functional_category_annotation.tsv",
        "data/sample_level_provenance.tsv",
        "scripts/run_comprehensive_batch1_single_omics.py",
        "scripts/run_comprehensive_triple_integration.py",
        "scripts/run_comprehensive_pairwise_integration.py",
        "scripts/run_final_9model_stacking.py",
        "scripts/run_advanced_method*.py",
        "scripts/run_interpretability_key_factors.py",
        "scripts/run_scheme2_category_masking.py",
        "scripts/run_other_omics_category_masking.py",
        "scripts/run_mrna_equal_mask_control.py",
        "scripts/supplement_masking_analyses.py",
        "scripts/generate_nsre_images.py",
        "scripts/adaptive_nsre.py",
        "scripts/pam50_gene_exclusion_sensitivity.py",
        "scripts/run_dense_baseline.py",
        "scripts/run_random_permutation_control.py",
        "scripts/run_stress_pathway_analysis.py",
        "scripts/generate_annotation_mapping.py",
        "scripts/generate_sample_provenance.py",
    ]:
        add_bullet(doc, item)
    add_body(
        doc,
        "为避免把大量结果表全部嵌入 Word，本补充材料仅保留关键汇总表与图示；全部 `data/**/*.tsv` 结果表已转换为 CSV 并打包为 `data/supplementary_data/supplementary_data_tables.zip`。压缩包内含 `manifest.tsv`，逐行记录源 TSV 的相对路径、对应 CSV 文件名、行数、列数与转换状态，作为机器可读结果的权威清单。读者可从 GitHub 仓库下载该压缩包逐项复核。",
    )
    add_body(
        doc,
        "软件与库版本：Python 3.11、scikit-learn、PyTorch、lifelines、pandas、NumPy 和 matplotlib。主要计算库引用为 scikit-learn[S16]、PyTorch[S17] 与 lifelines[S18]。",
    )
    add_body(
        doc,
        "复现时建议固定随机种子 42；所有特征筛选、标准化、数据增强和模型拟合均在交叉验证训练折内完成，测试折不参与任何预处理与参数选择。",
    )

    add_heading(doc, "S19. 补充材料参考文献", 1)
    add_body(doc, "[S1] Grossman RL, Heath AP, Ferretti V, Varmus HE, Lowy DR, Kibbe WA, et al. Toward a Shared Vision for Cancer Genomic Data. N Engl J Med. 2016;375(12):1109-1112. doi:10.1056/NEJMp1607591. PMID:27653561.")
    add_body(doc, "[S2] Goldman MJ, Craft B, Hastie M, Repecka K, McDade F, Kamath A, et al. Visualizing and interpreting cancer genomics data via the Xena platform. Nat Biotechnol. 2020;38(6):675-678. doi:10.1038/s41587-020-0546-8. PMID:32444850.")
    add_body(doc, "[S3] Mermel CH, Schumacher SE, Hill B, Meyerson ML, Beroukhim R, Getz G. GISTIC2.0 facilitates sensitive and confident localization of the targets of focal somatic copy-number alteration in human cancers. Genome Biol. 2011;12(4):R41. doi:10.1186/gb-2011-12-4-r41. PMID:21527027.")
    add_body(doc, "[S4] Guyon I, Elisseeff A. An introduction to variable and feature selection. J Mach Learn Res. 2003;3:1157-1182.")
    add_body(doc, "[S5] Tibshirani R. Regression shrinkage and selection via the lasso. J R Stat Soc Series B Stat Methodol. 1996;58(1):267-288. doi:10.1111/j.2517-6161.1996.tb02080.x.")
    add_body(doc, "[S6] Goodfellow I, Pouget-Abadie J, Mirza M, Xu B, Warde-Farley D, Ozair S, et al. Generative adversarial nets. In: Advances in Neural Information Processing Systems. 2014;27:2672-2680. arXiv:1406.2661.")
    add_body(doc, "[S7] Arjovsky M, Chintala S, Bottou L. Wasserstein generative adversarial networks. In: Proceedings of the 34th International Conference on Machine Learning. PMLR. 2017;70:214-223.")
    add_body(doc, "[S8] Gulrajani I, Ahmed F, Arjovsky M, Dumoulin V, Courville A. Improved training of Wasserstein GANs. In: Advances in Neural Information Processing Systems. 2017:5767-5777. arXiv:1704.00028.")
    add_body(doc, "[S9] Simonyan K, Vedaldi A, Zisserman A. Deep inside convolutional networks: visualising image classification models and saliency maps. arXiv:1312.6034. 2014.")
    add_body(doc, "[S10] Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE International Conference on Computer Vision. 2017:618-626. doi:10.1109/ICCV.2017.74.")
    add_body(doc, "[S11] Lundberg SM, Lee SI. A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems. 2017;30:4765-4774. arXiv:1705.07874.")
    add_body(doc, "[S12] Kozomara A, Birgaoanu M, Griffiths-Jones S. miRBase: from microRNA sequences to function. Nucleic Acids Res. 2019;47(D1):D155-D162. doi:10.1093/nar/gky1141. PMID:30423142.")
    add_body(doc, "[S13] The Gene Ontology Consortium. The Gene Ontology knowledgebase in 2023. Genetics. 2023;224(1):iyad031. doi:10.1093/genetics/iyad031. PMID:36866529.")
    add_body(doc, "[S14] Kanehisa M, Furumichi M, Sato Y, Ishiguro-Watanabe M, Tanabe M. KEGG: integrating viruses and cellular organisms. Nucleic Acids Res. 2021;49(D1):D545-D551. doi:10.1093/nar/gkaa970. PMID:33125081.")
    add_body(doc, "[S15] Gillespie M, Jassal B, Stephan R, Milacic M, Rothfels K, Senff-Ribeiro A, et al. The reactome pathway knowledgebase 2022. Nucleic Acids Res. 2022;50(D1):D687-D692. doi:10.1093/nar/gkab1028. PMID:34788843.")
    add_body(doc, "[S16] Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: Machine Learning in Python. J Mach Learn Res. 2011;12:2825-2830.")
    add_body(doc, "[S17] Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, et al. PyTorch: an imperative style, high-performance deep learning library. In: Advances in Neural Information Processing Systems. 2019;32:8024-8035.")
    add_body(doc, "[S18] Davidson-Pilon C. lifelines: survival analysis in Python. J Open Source Softw. 2019;4(40):1317. doi:10.21105/joss.01317.")

    doc.save(OUT_SUPP)
    print(OUT_SUPP)
    return doc


def main():
    build_main()
    build_supplement()
    export_supplementary_tables()


if __name__ == "__main__":
    main()
