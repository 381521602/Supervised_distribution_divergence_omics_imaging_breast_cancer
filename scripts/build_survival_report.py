#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build survival feature-selection report."""

from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "data" / "figures"
OUT = ROOT / "data" / "生存预测特征提取筛选_学术论文版.docx"
BLUE = RGBColor(0x2E,0x74,0xB5)


def set_cell_shading(cell, fill):
    tc_pr=cell._tc.get_or_add_tcPr(); shd=tc_pr.find(qn("w:shd"))
    if shd is None:
        shd=OxmlElement("w:shd"); tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_table_widths(table, widths):
    table.autofit=False
    for row in table.rows:
        for idx,w in enumerate(widths):
            row.cells[idx].width=Inches(w/1440)


def add_heading(doc,text,level=1):
    h=doc.add_heading(level=level); r=h.add_run(text)
    r.font.name="Calibri"; r.font.size=Pt(16 if level==1 else 13); r.font.bold=True; r.font.color.rgb=BLUE
    h.paragraph_format.space_before=Pt(12); h.paragraph_format.space_after=Pt(6)


def add_body(doc,text):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; r=p.add_run(text); r.font.name="Calibri"; r.font.size=Pt(11)
    p.paragraph_format.space_after=Pt(6); p.paragraph_format.line_spacing=1.2


def add_equation(doc, omml):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p._p.append(parse_xml(omml)); p.paragraph_format.space_after=Pt(6)


def add_figure(doc,path,caption):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; run=p.add_run(); run.add_picture(str(path),width=Inches(6.0))
    c=doc.add_paragraph(); c.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=c.add_run(caption); r.font.name="Calibri"; r.font.size=Pt(9); r.font.color.rgb=RGBColor(0x55,0x55,0x55)


def main():
    doc=Document(); sec=doc.sections[0]
    sec.page_width=Inches(8.5); sec.page_height=Inches(11)
    sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Inches(1)
    normal=doc.styles["Normal"]; normal.font.name="Calibri"; normal.font.size=Pt(11)

    title=doc.add_paragraph(); title.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=title.add_run("乳腺癌生存预测的多组学特征提取筛选研究"); r.font.name="Calibri"; r.font.size=Pt(17); r.font.bold=True

    add_heading(doc,"摘要",1)
    add_body(doc,"本研究在 TCGA-BRCA 生存预测任务上，系统比较了低方差过滤、F 值、L1、JSD 及其两两组合等特征提取方法，以及逻辑回归、随机森林、梯度提升、支持向量机、K 近邻和 Cox 比例风险模型。结果表明，mRNA 在 V_JSD_F 方法下 Cox C-index 达到 0.6604；CNV 在 V_L1_JSD 方法下 C-index 为 0.6124；miRNA 在 V_L1_JSD 方法下 C-index 为 0.5915。生存二分类 AUC 方面，mRNA 使用 V_L1_F+LogisticRegression 达到 0.6586。")

    add_heading(doc,"1. 引言",1)
    add_body(doc,"生存预测是乳腺癌精准医疗的重要内容。组学数据具有高维、小样本和异质性，特征提取和模型选择对生存预测性能影响显著。本文对三组学分别进行特征提取和生存建模，比较不同方法组合的表现。")
    add_heading(doc,"1.1 数据集构建",2)
    add_body(doc,"生存预测最终数据集使用 TCGA-BRCA 的 mRNA、CNV、miRNA 数据。特征提取方案为：mRNA 采用 V_JSD_F 保留 200 特征，CNV 采用 V_L1_JSD 保留 150 特征，miRNA 采用 V_L1_JSD 保留 600 特征。样本量分别为 1073、1057、739。")
    table=doc.add_table(rows=1, cols=4)
    hdr=table.rows[0].cells
    for i,txt in enumerate(["组学","特征方法","特征数","样本数"]):
        hdr[i].text=txt; set_cell_shading(hdr[i],"F2F4F7")
    for row in [("mRNA","V_JSD_F","200","1073"),("CNV","V_L1_JSD","150","1057"),("miRNA","V_L1_JSD","600","739")]:
        cells=table.add_row().cells
        for i,txt in enumerate(row): cells[i].text=txt
    set_table_widths(table,[1800,2200,1800,1800])

    add_heading(doc,"2. 方法",1)
    add_heading(doc,"2.1 特征提取算法",2)
    add_body(doc,"低方差过滤：Var(x_j)<τ 的特征被删除，τ=0。F 值：F=MS_between/MS_within，选择 F 最大的 k 个特征。L1：min_w (1/2)||Xw-y||^2 + λ||w||_1。JSD：JSD(p||q)=Σ p_i log2(2p_i/(p_i+q_i)) + Σ q_i log2(2q_i/(p_i+q_i))。")
    add_equation(doc, '<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath><m:f><m:num><m:r><m:t>MS_between</m:t></m:r></m:num><m:den><m:r><m:t>MS_within</m:t></m:r></m:den></m:f></m:oMath></m:oMathPara>')
    add_equation(doc, '<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath><m:r><m:t>JSD(p||q)=Σ p_i log2(2p_i/(p_i+q_i)) + Σ q_i log2(2q_i/(p_i+q_i))</m:t></m:r></m:oMath></m:oMathPara>')
    add_heading(doc,"2.2 机器学习模型",2)
    add_body(doc,"生存二分类使用 LogisticRegression、RandomForest、GradientBoosting、SVC、KNN，以 ROC AUC 为评价指标。生存时间分析使用 Cox 比例风险模型，风险函数 h(t|x)=h0(t)exp(β^T x)，以 C-index 为评价指标。")
    add_equation(doc, '<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath><m:r><m:t>h(t|x)=h0(t)exp(β^T x)</m:t></m:r></m:oMath></m:oMathPara>')
    add_heading(doc,"2.3 实验设计与组合总数",2)
    table=doc.add_table(rows=1, cols=3)
    hdr=table.rows[0].cells
    for i,txt in enumerate(["项目","设置","说明"]):
        hdr[i].text=txt; set_cell_shading(hdr[i],"F2F4F7")
    for row in [("组学","mRNA / CNV / miRNA","分别分析"),("基础特征数","200 / 50 / 200","分别扩增 1-5 倍"),("特征方法","10 种","V 及九种组合"),("生存模型","LR/RF/GB/SVC/KNN/CoxPH","AUC 与 C-index"),("交叉验证","5 折分层","random_state=42")]:
        cells=table.add_row().cells
        for i,txt in enumerate(row): cells[i].text=txt
    set_table_widths(table,[1600,3600,4160])
    add_body(doc,"综上，生存预测实验共包含 10 种特征方法 × 3 种组学 × 5 个特征倍数 × 6 种生存模型 = 900 组 AUC 组合，以及 10 × 3 × 5 = 150 组 CoxPH 组合。")

    add_heading(doc,"3. 结果",1)
    add_body(doc,"生存二分类 AUC 最优组合：mRNA 使用 V_L1_F+LogisticRegression、800 特征时 AUC=0.6586；CNV 使用 V_L1_F+RandomForest、50 特征时 AUC=0.5559；miRNA 使用 V_JSD+RandomForest、200 特征时 AUC=0.6046。")
    add_body(doc,"Cox C-index 最优组合：mRNA 使用 V_JSD_F、200 特征时 C-index=0.6604；CNV 使用 V_L1_JSD、150 特征时 C-index=0.6124；miRNA 使用 V_L1_JSD、800 特征时 C-index=0.5915。")
    add_figure(doc, FIG/"survival_best_methods_auc.png", "图 1. 生存 AUC 最优特征方法在不同特征数下的表现")
    add_figure(doc, FIG/"survival_best_methods_cox.png", "图 2. Cox C-index 最优特征方法在不同特征数下的表现")

    add_heading(doc,"4. 讨论",1)
    add_body(doc,"mRNA 在生存预测中表现最好，C-index 可达 0.66；CNV 和 miRNA 较弱。不同组学适合的特征方法和特征数量不同：mRNA 使用 JSD 相关组合，CNV 和 miRNA 使用 L1_JSD 组合。")

    add_heading(doc,"4.1 综合效率后的最优组合",2)
    table=doc.add_table(rows=1, cols=5)
    hdr=table.rows[0].cells
    for i,txt in enumerate(["组学","特征方法","特征数","C-index","选择理由"]):
        hdr[i].text=txt; set_cell_shading(hdr[i],"F2F4F7")
    rows=[
        ("mRNA","V_JSD_F","200","0.6604","200 特征已最优，效率最高"),
        ("CNV","V_L1_JSD","150","0.6124","150 为拐点，200 特征收益很小"),
        ("miRNA","V_L1_JSD","600","0.5862","600 接近 800 的最优值，计算更高效"),
    ]
    for row in rows:
        cells=table.add_row().cells
        for i,txt in enumerate(row): cells[i].text=txt
    set_table_widths(table,[1100,1800,1100,1500,2860])

    add_heading(doc,"5. 结论",1)
    add_body(doc,"本研究为乳腺癌生存预测提供了系统的特征提取和模型选择依据。综合考虑预测精度与计算效率，推荐 mRNA 采用 V_JSD_F、200 特征；CNV 采用 V_L1_JSD、150 特征；miRNA 采用 V_L1_JSD、600 特征。mRNA 是生存预测的主要组学。")

    doc.save(OUT); print(f"Saved -> {OUT}")


if __name__=="__main__":
    main()
