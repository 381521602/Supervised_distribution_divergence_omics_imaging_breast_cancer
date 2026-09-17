#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build detailed PAM50 feature-selection report."""

from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "data" / "figures"
OUT = ROOT / "data" / "PAM50特征提取筛选详细报告.docx"
BLUE = RGBColor(0x2E,0x74,0xB5)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr(); shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd"); tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_table_widths(table, widths):
    table.autofit = False
    for row in table.rows:
        for idx, w in enumerate(widths):
            row.cells[idx].width = Inches(w/1440)


def add_heading(doc, text, level=1):
    h=doc.add_heading(level=level); r=h.add_run(text)
    r.font.name="Calibri"; r.font.size=Pt(16 if level==1 else 13); r.font.bold=True; r.font.color.rgb=BLUE
    h.paragraph_format.space_before=Pt(12); h.paragraph_format.space_after=Pt(6)


def add_body(doc, text):
    p=doc.add_paragraph(); r=p.add_run(text); r.font.name="Calibri"; r.font.size=Pt(11)
    p.paragraph_format.space_after=Pt(6); p.paragraph_format.line_spacing=1.15


def add_figure(doc, path, caption):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; run=p.add_run(); run.add_picture(str(path), width=Inches(6.0))
    c=doc.add_paragraph(); c.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=c.add_run(caption); r.font.name="Calibri"; r.font.size=Pt(9); r.font.color.rgb=RGBColor(0x55,0x55,0x55)


def main():
    doc=Document(); sec=doc.sections[0]
    sec.page_width=Inches(8.5); sec.page_height=Inches(11)
    sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Inches(1)
    normal=doc.styles["Normal"]; normal.font.name="Calibri"; normal.font.size=Pt(11)

    t=doc.add_paragraph(); t.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=t.add_run("PAM50 分型特征提取筛选详细报告"); r.font.name="Calibri"; r.font.size=Pt(18); r.font.bold=True

    add_heading(doc,"1. 特征提取算法",1)
    add_body(doc,"低方差过滤：Var(x_j)<τ 的特征被删除，τ=0。F 值：F=MS_between/MS_within，选择 F 最大的 k 个特征。L1：min_w (1/2)||Xw-y||^2 + λ||w||_1，利用稀疏性自动选择特征。JSD：JSD(p||q)=Σ p_i log2(2p_i/(p_i+q_i)) + Σ q_i log2(2q_i/(p_i+q_i))，度量两类特征分布距离。")
    add_body(doc,"两两组合包括：V_F、V_L1、V_JSD、V_F_L1、V_F_JSD、V_L1_F、V_L1_JSD、V_JSD_F、V_JSD_L1。")

    add_heading(doc,"2. 机器学习算法",1)
    add_body(doc,"LogisticRegression：L2 正则，max_iter=2000。RandomForest：n_estimators=100。GradientBoosting：n_estimators=100。SVC：线性核。KNN：k=5。MLP：输入-128-ReLU-Dropout(0.3)-64-ReLU-Dropout(0.3)-输出，Adam(lr=0.001, weight_decay=0.0001)，15 个 epoch。")

    add_heading(doc,"3. 实验设计",1)
    add_body(doc,"mRNA、CNV、miRNA 基础特征数分别为 200、50、200，并按 2、3、4、5 倍扩增。所有组合采用 5 折分层交叉验证，随机种子 42，记录均值与标准差。")

    add_heading(doc,"4. 结果分析",1)
    add_figure(doc, FIG/"pam50_best_methods_by_k.png", "图 1. 各组学最优特征方法在不同特征数下的 Accuracy（mean ± std）")
    add_figure(doc, FIG/"pam50_lr_vs_mlp.png", "图 2. LogisticRegression 与 MLP 最优 Accuracy 对比")

    add_heading(doc,"4.1 各组学最优组合",2)
    table = doc.add_table(rows=1, cols=6)
    hdr=table.rows[0].cells
    for i,txt in enumerate(["组学","特征数","倍数","特征方法","模型","Accuracy"]):
        hdr[i].text=txt; set_cell_shading(hdr[i],"F2F4F7")
    rows=[
        ("mRNA","400","2×","V_L1","MLP","0.9136"),
        ("miRNA","600","3×","V_F_JSD","MLP","0.8471"),
        ("CNV","50","1×","V_L1_F","MLP","0.7078"),
    ]
    for row in rows:
        cells=table.add_row().cells
        for i,txt in enumerate(row): cells[i].text=txt
    set_table_widths(table,[1000,1000,1000,1800,1200,1200])

    add_heading(doc,"5. 讨论与结论",1)
    add_body(doc,"mRNA 在 V_L1+MLP、400 特征时即可达到约 0.914 的 Accuracy，继续增加到 1000 特征收益很小。miRNA 在 600 特征时达到拐点。CNV 保持 50 特征即可。MLP 在各组学上均优于传统机器学习模型。")

    doc.save(OUT); print(f"Saved -> {OUT}")


if __name__=="__main__":
    main()
