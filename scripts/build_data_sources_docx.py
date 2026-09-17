#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a detailed data-sources Word document."""

from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "data" / "figures"
OUT = ROOT / "data" / "数据来源详细说明.docx"
BLUE = RGBColor(0x2E, 0x74, 0xB5)
DARK = RGBColor(0x1F, 0x4D, 0x78)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd"); tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_table_widths(table, widths):
    table.autofit = False
    for row in table.rows:
        for idx, w in enumerate(widths):
            row.cells[idx].width = Inches(w / 1440)


def add_heading(doc, text, level=1):
    h = doc.add_heading(level=level)
    r = h.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(16 if level == 1 else 13)
    r.font.bold = True
    r.font.color.rgb = BLUE if level == 1 else DARK
    h.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    h.paragraph_format.space_after = Pt(6 if level == 1 else 4)
    return h


def add_body(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Calibri"; r.font.size = Pt(11)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.2
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(item)
        r.font.name = "Calibri"; r.font.size = Pt(11)
        p.paragraph_format.space_after = Pt(3)


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    hdr = table.rows[0].cells
    for idx, text in enumerate(headers):
        hdr[idx].text = text
        set_cell_shading(hdr[idx], "F2F4F7")
        for run in hdr[idx].paragraphs[0].runs:
            run.font.bold = True; run.font.name = "Calibri"; run.font.size = Pt(10.5)
    for row in rows:
        cells = table.add_row().cells
        for idx, text in enumerate(row):
            cells[idx].text = str(text)
            for run in cells[idx].paragraphs[0].runs:
                run.font.name = "Calibri"; run.font.size = Pt(10.5)
    set_table_widths(table, widths)
    return table


def add_figure(doc, path, caption):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(); run.add_picture(str(path), width=Inches(6.0))
    c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption); r.font.name = "Calibri"; r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x55,0x55,0x55)


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.5); sec.page_height = Inches(11)
    sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(1)
    normal = doc.styles["Normal"]; normal.font.name = "Calibri"; normal.font.size = Pt(11)

    title = doc.add_paragraph(); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("TCGA-BRCA 多组学数据来源详细说明"); r.font.name = "Calibri"; r.font.size = Pt(18); r.font.bold = True; r.font.color.rgb = RGBColor(0x0B,0x25,0x45)

    add_heading(doc, "1. 数据来源与伦理声明", 1)
    add_body(doc, "本研究使用的乳腺癌多组学数据均来自公开数据库 The Cancer Genome Atlas（TCGA）的 BRCA 项目（TCGA-BRCA）。数据通过两个公开平台获取：Genomic Data Commons（GDC）用于临床、随访、样本元数据和文件清单；UCSC Xena 用于获取已标准化、可直接用于建模的组学表达矩阵和临床矩阵。TCGA 数据为公开、去标识化的人类肿瘤研究数据，本研究不涉及新的患者样本采集，符合相关伦理与数据使用规范。")

    add_heading(doc, "2. 数据平台与访问", 1)
    add_bullets(doc, [
        "GDC Data Portal：https://portal.gdc.cancer.gov/projects/TCGA-BRCA",
        "UCSC Xena：https://xenabrowser.net/datapages/?cohort=TCGA%20Breast%20Cancer%20(BRCA)",
        "所有数据均为 open access，无需额外授权。"
    ])

    add_heading(doc, "3. 组学数据类型与预处理", 1)
    add_table(
        doc,
        ["组学", "数据集", "数据格式与预处理"],
        [
            ["mRNA", "UCSC Xena: TCGA.BRCA.sampleMap/HiSeqV2", "基因级表达矩阵，log2 转换的 RSEM 归一化表达值"],
            ["CNV", "UCSC Xena: Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes", "GISTIC2 基因级阈值化拷贝数，取值为 -2, -1, 0, 1, 2"],
            ["miRNA", "UCSC Xena: TCGA.BRCA.sampleMap/miRNA_HiSeq_gene", "miRNA 表达矩阵"],
            ["临床", "UCSC Xena: BRCA_clinicalMatrix；GDC 临床/随访", "生存、分期、治疗、复发、受体状态等"],
            ["分子分型", "BRCA_clinicalMatrix 中的 PAM50 字段", "PAM50Call_RNAseq、PAM50_mRNA_nature2012"],
        ],
        [1300, 4700, 3360],
    )

    add_heading(doc, "4. 样本纳入与样本量", 1)
    add_body(doc, "初始纳入 TCGA-BRCA 共 1098 例，其中同时具有 mRNA、miRNA 和基因级 CNV 三类组学数据的病例为 1073 例。样本纳入标准为：具有对应组学数据，且相应结局标签非缺失。")
    add_table(
        doc,
        ["任务", "mRNA", "CNV", "miRNA", "三组学交集"],
        [
            ["PAM50 四分类", "833", "818", "497", "493"],
            ["生存分析", "1073", "1057", "739", "731"],
        ],
        [2000, 1600, 1600, 1600, 1800],
    )
    add_figure(doc, FIG / "fig1_data_coverage.png", "图 1. 各组学可用样本数及三组学重叠样本数")

    add_heading(doc, "5. 结局定义", 1)
    add_heading(doc, "5.1 分子分型", 2)
    add_body(doc, "采用 PAM50 分子分型，四分类为 Luminal A、Luminal B、Basal-like、HER2-enriched。Normal-like 样本被剔除，以保证类别一致性。")
    add_heading(doc, "5.2 总生存期", 2)
    add_body(doc, "生存时间定义为从诊断到死亡或末次随访的时间。死亡事件由 vital_status=Dead 定义，并使用 days_to_death；删失样本使用末次随访时间。生存事件编码：1=死亡，0=删失。")
    add_heading(doc, "5.3 受体状态", 2)
    add_body(doc, "ER、PR、HER2 状态来自临床矩阵，仅保留 Positive/Negative；Indeterminate、Equivocal 作为缺失处理。")

    add_heading(doc, "6. 数据预处理流程", 1)
    add_bullets(doc, [
        "按 TCGA barcode 对三组学样本进行对齐，优先选择原发肿瘤样本。",
        "对每组学分别进行特征筛选：低方差过滤、F 值、L1、JSD 及其组合。",
        "建模前在交叉验证训练折内部进行标准化，避免信息泄漏。",
        "最终用于图像化模型的特征数：mRNA 200、CNV 50、miRNA 200。",
    ])

    add_heading(doc, "7. 数据文件与可复现性", 1)
    add_body(doc, "本文使用的样本清单、标签表和下载脚本已随项目保存，确保结果可复现。")

    doc.save(OUT)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
