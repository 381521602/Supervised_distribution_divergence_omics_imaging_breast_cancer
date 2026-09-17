#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the stage-summary Word document."""

from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "data" / "figures"
OUT = ROOT / "data" / "TCGA-BRCA_阶段总结.docx"

BLUE = RGBColor(0x2E, 0x74, 0xB5)
DARK_BLUE = RGBColor(0x1F, 0x4D, 0x78)
INK = RGBColor(0x0B, 0x25, 0x45)
GRAY_FILL = "F2F4F7"


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, bottom=80, start=120, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.find(qn("w:tcMar"))
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, val in (("top", top), ("bottom", bottom), ("start", start), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")


def set_table_widths(table, widths_dxa):
    table.autofit = False
    tbl = table._tbl
    tbl_pr = tbl.tblPr

    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths_dxa)))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_w.addnext(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")

    grid = tbl.find(qn("w:tblGrid"))
    if grid is not None:
        grid_cols = grid.findall(qn("w:gridCol"))
        for idx, col in enumerate(grid_cols):
            if idx < len(widths_dxa):
                col.set(qn("w:w"), str(widths_dxa[idx]))

    for row in table.rows:
        for idx, width in enumerate(widths_dxa):
            row.cells[idx].width = Inches(width / 1440)


def add_heading(doc, text, level):
    h = doc.add_heading(level=level)
    run = h.add_run(text)
    if level == 1:
        run.font.name = "Calibri"
        run.font.size = Pt(16)
        run.font.color.rgb = BLUE
        run.font.bold = True
        h.paragraph_format.space_before = Pt(16)
        h.paragraph_format.space_after = Pt(8)
    elif level == 2:
        run.font.name = "Calibri"
        run.font.size = Pt(13)
        run.font.color.rgb = BLUE
        run.font.bold = True
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(6)
    else:
        run.font.name = "Calibri"
        run.font.size = Pt(12)
        run.font.color.rgb = DARK_BLUE
        run.font.bold = True
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(4)
    return h


def add_body(doc, text, bold=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run.bold = bold
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.10
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(item)
        run.font.name = "Calibri"
        run.font.size = Pt(11)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.10


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    hdr = table.rows[0].cells
    for idx, text in enumerate(headers):
        hdr[idx].text = text
        set_cell_shading(hdr[idx], GRAY_FILL)
        set_cell_margins(hdr[idx])
        for run in hdr[idx].paragraphs[0].runs:
            run.font.bold = True
            run.font.name = "Calibri"
            run.font.size = Pt(10.5)
    for row in rows:
        cells = table.add_row().cells
        for idx, text in enumerate(row):
            cells[idx].text = str(text)
            set_cell_margins(cells[idx])
            for run in cells[idx].paragraphs[0].runs:
                run.font.name = "Calibri"
                run.font.size = Pt(10.5)
    set_table_widths(table, widths)
    return table


def add_figure(doc, path, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(6.2))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.font.name = "Calibri"
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    cap.paragraph_format.space_before = Pt(2)
    cap.paragraph_format.space_after = Pt(10)


def main() -> None:
    doc = Document()

    # Page geometry: US Letter, 1 inch margins.
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.orientation = WD_ORIENT.PORTRAIT
    section.top_margin = Inches(1)
    section.right_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)

    # Base style.
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    # Title.
    title = doc.add_paragraph()
    title.paragraph_format.space_after = Pt(8)
    tr = title.add_run("TCGA-BRCA 组学数据图像化项目阶段总结")
    tr.font.name = "Calibri"
    tr.font.size = Pt(20)
    tr.font.bold = True
    tr.font.color.rgb = INK

    add_body(doc, "本报告总结目前已完成的数据模块、特征提取模块和单组学预测模块，用于后续多组学融合分析。")

    # 一、数据模块
    add_heading(doc, "一、数据模块", 1)
    add_body(doc, "已从 GDC API 和 UCSC Xena 收集 TCGA-BRCA 的 mRNA、CNV、miRNA 三组学数据，并完成样本对齐和标签整理。")
    add_bullets(doc, [
        "共 1098 例乳腺癌病例，其中 1073 例同时具备 mRNA、miRNA 和基因级 CNV 三类数据。",
        "临床标签覆盖 PAM50 分型、ER/PR/HER2、TNBC、生存时间/事件、分期、淋巴结、复发和新发肿瘤事件。",
        "已生成建模就绪标签表、统一主清单和文件级下载清单。",
    ])
    add_table(
        doc,
        ["文件", "内容"],
        [
            ["brca_master_table.tsv", "统一主清单：临床标签、三组学文件/样本编号、PAM50、ER/PR/HER2"],
            ["brca_labels_modeling_ready.tsv", "建模就绪标签表，含清洗后的分类、生存和复发标签"],
            ["brca_clinical.tsv", "病例级 GDC 临床与随访标签"],
            ["brca_molecular_subtype.tsv", "PAM50 和 ER/PR/HER2 状态"],
            ["brca_files.tsv", "文件级下载清单"],
        ],
        [2700, 6660],
    )
    add_figure(doc, FIG / "fig1_data_coverage.png", "图 1. 各组学可用样本数及三组学重叠样本数")

    # 二、特征提取模块
    add_heading(doc, "二、特征提取模块", 1)
    add_body(doc, "已实现多种传统特征筛选/提取方法，以及 JSD 新对称相对熵筛选、One-vs-Rest JSD 和两阶段组合方法。")
    add_table(
        doc,
        ["组学", "PAM50 分型最优方法", "说明"],
        [
            ["mRNA", "FClassif → JSD", "准确率最高，约 0.909"],
            ["CNV", "L1 → One-vs-Rest JSD", "准确率 0.674，优于 PCA"],
            ["miRNA", "JSD → L1", "准确率 0.825"],
        ],
        [1400, 3000, 4960],
    )
    add_figure(doc, FIG / "fig4_adaptive_pipeline.png", "图 2. 自适应 JSD 特征筛选流程")
    add_body(doc, "自适应模块已固化，并导出每个组学、每个任务对应的“样本 × 200 个筛选特征”矩阵，可直接用于多组学融合。")

    # 三、单组学预测模块
    add_heading(doc, "三、单组学预测模块", 1)
    add_body(doc, "采用经典机器学习方法，对 PAM50 四分类和总生存期进行 5 折交叉验证。")

    add_heading(doc, "PAM50 四分类结果", 2)
    add_table(
        doc,
        ["组学", "最优方法", "Accuracy", "Macro-F1"],
        [
            ["mRNA", "FClassif_JSD", "0.9087", "0.8973"],
            ["CNV", "L1_OvRJSD", "0.6736", "0.6133"],
            ["miRNA", "JSD_L1", "0.8250", "0.7808"],
        ],
        [2000, 3360, 2000, 2000],
    )
    add_figure(doc, FIG / "fig2_pam50_results.png", "图 3. 单组学 PAM50 四分类最优结果")

    add_heading(doc, "总生存期结果", 2)
    add_table(
        doc,
        ["组学", "任务", "最优方法", "指标"],
        [
            ["mRNA", "生存二分类", "L1_OvRJSD", "AUC 0.6183"],
            ["mRNA", "生存 Cox", "JSD", "C-index 0.6339"],
            ["CNV", "生存 Cox", "L1_JSD", "C-index 0.5640"],
            ["miRNA", "生存 Cox", "JSD", "C-index 0.5048"],
        ],
        [1800, 2500, 2600, 2460],
    )
    add_figure(doc, FIG / "fig3_survival_results.png", "图 4. 单组学生存预测最优结果")

    add_heading(doc, "关键结论", 2)
    add_bullets(doc, [
        "mRNA 是 PAM50 分型的主信号，单组学即可达到约 91% 准确率。",
        "CNV 单独较弱，但通过 L1 + One-vs-Rest JSD 后有明显改善。",
        "miRNA 中等，最优方法约 82% 准确率。",
        "生存预测单组学偏弱，为后续多组学融合提供了提升空间。",
    ])

    add_heading(doc, "四、多组学融合模块", 1)
    add_body(doc, "在统一筛选特征的基础上，测试了三组学简单拼接、两两融合，以及保留全样本的缺失组学处理。")
    add_table(
        doc,
        ["融合组合", "PAM50 最优 Accuracy / Macro-F1", "生存 Cox C-index"],
        [
            ["mRNA+CNV", "0.8533 / 0.8198", "0.5440"],
            ["mRNA+miRNA", "0.8329 / 0.8046", "0.5674"],
            ["CNV+miRNA", "0.7546 / 0.6970", "0.6053"],
            ["三组学拼接", "0.8438 / 0.8113", "0.5434"],
        ],
        [2800, 3760, 2800],
    )
    add_figure(doc, FIG / "fig5_pam50_fusion_dl.png", "图 5. PAM50 分型：单组学、融合与深度学习对比")
    add_figure(doc, FIG / "fig6_survival_fusion_dl.png", "图 6. 生存预测：单组学、融合与深度学习对比")

    add_heading(doc, "五、深度学习模块", 1)
    add_body(doc, "使用 PyTorch 训练了简单 MLP 模型，在同样本和同特征下与经典机器学习对比。")
    add_table(
        doc,
        ["任务", "传统最优", "深度学习 MLP"],
        [
            ["PAM50 mRNA+CNV", "0.8533 / 0.8198", "0.8533 / 0.8318"],
            ["生存 Cox mRNA+CNV", "0.5440", "0.6043"],
            ["生存 Cox 三组学拼接", "0.5434", "0.6053"],
        ],
        [3400, 3000, 2960],
    )

    add_heading(doc, "六、下一步", 1)
    add_body(doc, "三组学筛选后的特征矩阵已准备就绪，下一步进入多组学融合预测模块，重点比较简单拼接、早期融合、晚期融合和注意力融合，观察 PAM50 分型与生存 C-index 是否相较单组学提升。")

    out_v2 = ROOT / "data" / "TCGA-BRCA_阶段总结_完整版.docx"
    doc.save(out_v2)
    print(f"Saved -> {out_v2}")


if __name__ == "__main__":
    main()
