#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build comprehensive multi-omics integration DOCX report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = DATA / "images/comprehensive_report"
OUT = ROOT / "多组学整合综合分析报告.docx"


def read(name):
    return pd.read_csv(DATA / name, sep="\t")


def fmt(row):
    return f"{row['mean']:.4f} ± {row['std']:.4f}"


def set_style(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25
    for name, size, color, before, after in [
        ("Heading 1", 16, RGBColor(0x2E, 0x74, 0xB5), 18, 10),
        ("Heading 2", 13, RGBColor(0x2E, 0x74, 0xB5), 14, 7),
        ("Heading 3", 12, RGBColor(0x1F, 0x4D, 0x78), 10, 5),
    ]:
        st = doc.styles[name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.color.rgb = color
        st.font.bold = True
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)


def add_title(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    run.font.size = Pt(20)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1F, 0x4D, 0x78)
    p.paragraph_format.space_after = Pt(10)


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    p.paragraph_format.space_after = Pt(8)


def add_description(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    p.paragraph_format.space_after = Pt(8)


def set_table_width(table, widths):
    table.autofit = False
    tbl = table._tbl
    tblPr = tbl.tblPr
    for el in tblPr.findall(qn("w:tblW")):
        tblPr.remove(el)
    tblW = tblPr.makeelement(qn("w:tblW"), {qn("w:w"): str(int(sum(widths) * 1440)), qn("w:type"): "dxa"})
    tblPr.append(tblW)
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            if i < len(widths):
                cell.width = Inches(widths[i])


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    if widths is None:
        widths = [6.5 / len(headers)] * len(headers)
    set_table_width(table, widths)
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.font.bold = True
        run.font.size = Pt(9)
    for r in rows:
        cells = table.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(9)


def best_row(df, **filters):
    sub = df.copy()
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    return sub.sort_values("mean", ascending=False).iloc[0]


def rows_from(df, key_col, key_vals, metrics):
    rows = []
    for key in key_vals:
        row = [key]
        for m in metrics:
            sub = df[(df[key_col] == key) & (df["metric"] == m)]
            row.append(fmt(sub.iloc[0]) if not sub.empty else "")
        rows.append(row)
    return rows


def main():
    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(1); s.bottom_margin = Inches(1); s.left_margin = Inches(1); s.right_margin = Inches(1)
    set_style(doc)
    add_title(doc, "全面多组学整合综合分析报告")
    doc.add_paragraph("本报告汇总 ML、MLP、CNN 三类算法在三组学交集和两两组学交集上的整合结果。所有结果均基于预选特征，采用 5 折交叉验证，报告均值 ± 标准差。")

    doc.add_heading("1. 样本对齐与实验设计", level=1)
    doc.add_paragraph("三组学整合使用 mRNA∩CNV∩miRNA 共同样本；两两组学整合使用对应两组的交集。PAM50 三组学交集为 493 例，Survival 三组学交集为 731 例。")
    add_table(doc, ["任务", "mRNA", "CNV", "miRNA"], [
        ["PAM50 完整样本", "833", "818", "497"],
        ["Survival 完整样本", "1073", "1057", "739"],
    ], widths=[1.5, 1.5, 1.5, 1.5])
    add_description(doc, "表 1 为各组学完整样本数。单组学分析使用各自完整样本；整合分析使用相应交集样本。")
    add_table(doc, ["两两组合", "PAM50 交集", "Survival 交集"], [
        ["mRNA ∩ CNV", "818", "1055"],
        ["mRNA ∩ miRNA", "497", "737"],
        ["CNV ∩ miRNA", "493", "733"],
    ], widths=[1.5, 1.5, 1.5])
    add_description(doc, "表 2 为两两组学交集样本数。三组学交集为 PAM50 493 例、Survival 731 例。")
    doc.add_paragraph("基础算法包括传统 ML、MLP 和 FullSizeCNN；整合方式包括特征拼接、分支融合、晚期融合、Stacking、Transformer、DeepSurv、低秩双线性和简化图神经网络。")

    doc.add_heading("2. 三组学交集单组学基线", level=1)
    df = read("comprehensive_intersection_single_omics_results.tsv")
    best_pam = []
    for o in ["mRNA", "CNV", "miRNA"]:
        acc = best_row(df, omics=o, task="PAM50", metric="accuracy")
        f1 = best_row(df, omics=o, task="PAM50", metric="macro_f1")
        best_pam.append([o, acc["model"], fmt(acc), fmt(f1)])
    add_table(doc, ["组学", "最优模型", "Accuracy", "Macro-F1"], best_pam, widths=[0.8, 1.5, 2.0, 2.2])
    add_description(doc, "表 1 显示，在三组学交集样本上，mRNA 单组学 LogisticRegression 的 Accuracy 最高，为 0.9270；miRNA 单组学 LogisticRegression 为 0.8418；CNV 单组学 MLP 为 0.7324。mRNA 是 PAM50 分型的主要信号来源。")
    doc.add_picture(str(FIG / "fig_triple_single.png"), width=Inches(6.2))
    add_caption(doc, "图 1. 三组学交集单组学最优结果")
    add_description(doc, "图 1 左图展示 PAM50 分型，横轴为 mRNA、CNV、miRNA，纵轴为性能值，蓝色柱为 Accuracy，浅蓝色柱为 Macro-F1。mRNA 的 Accuracy 和 Macro-F1 均最高，且误差线较窄；CNV 和 miRNA 明显较低。右图展示生存预测，深蓝色柱为 ROC AUC，浅蓝色柱为 C-index。mRNA 仍最高，CNV 和 miRNA 较低。误差线表示 5 折交叉验证的标准差。")

    best_sur = []
    for o in ["mRNA", "CNV", "miRNA"]:
        auc = best_row(df, omics=o, task="Survival", metric="roc_auc")
        ci = best_row(df, omics=o, task="Survival", metric="c_index")
        best_sur.append([o, auc["model"], fmt(auc), fmt(ci)])
    add_table(doc, ["组学", "最优模型", "ROC AUC", "C-index"], best_sur, widths=[0.8, 1.5, 2.0, 2.2])
    add_description(doc, "表 2 显示，Survival 三组学交集单组学中，mRNA MLP/FullSizeCNN 表现最好，C-index 约 0.74；CNV LogisticRegression 次之；miRNA 较弱。")

    doc.add_heading("3. 三组学整合", level=1)
    df = read("comprehensive_triple_integration_results.tsv")
    schemes = ["Concat_LogisticRegression", "Concat_SVC", "Concat_MLP", "CNN_TripleConcat", "CNN_TripleGated", "CNN_LateAvg"]
    add_table(doc, ["方案", "PAM50 Accuracy", "PAM50 Macro-F1"], rows_from(df[df.task == "PAM50"], "scheme", schemes, ["accuracy", "macro_f1"]), widths=[1.6, 2.2, 2.2])
    add_table(doc, ["方案", "Survival ROC AUC", "Survival C-index"], rows_from(df[df.task == "Survival"], "scheme", schemes, ["roc_auc", "c_index"]), widths=[1.6, 2.2, 2.2])
    doc.add_picture(str(FIG / "fig_triple_integration.png"), width=Inches(6.2))
    add_caption(doc, "图 2. 三组学整合结果")
    add_description(doc, "PAM50 三组学整合中 Concat+SVC 的 Accuracy 最高，为 0.9168；Survival 三组学整合中 Concat+MLP 的 C-index 最高，为 0.7588，CNN 晚期平均的 ROC AUC 最高，为 0.7591。")
    add_description(doc, "图 2 左图比较 PAM50 各整合方案。蓝色柱为 Accuracy，浅蓝色柱为 Macro-F1；Concat+SVC 的 Accuracy 最高，CNN 分支和门控融合的误差线较大。右图比较生存预测，深蓝色柱为 ROC AUC，浅蓝色柱为 C-index；Concat+MLP 和 CNN 晚期平均表现最好，且 C-index 误差相对较小。")

    doc.add_heading("4. 两两组学整合", level=1)
    df = read("comprehensive_pairwise_integration_results.tsv")
    pair_rows = []
    for p in ["mRNA+CNV", "mRNA+miRNA", "CNV+miRNA"]:
        r = best_row(df, pair=p, task="PAM50", metric="accuracy")
        pair_rows.append([p, r["scheme"], fmt(r), fmt(best_row(df, pair=p, task="PAM50", metric="macro_f1"))])
    add_table(doc, ["组合", "最优方案", "Accuracy", "Macro-F1"], pair_rows, widths=[1.3, 1.7, 1.7, 1.8])
    pair_sur = []
    for p in ["mRNA+CNV", "mRNA+miRNA", "CNV+miRNA"]:
        r = best_row(df, pair=p, task="Survival", metric="roc_auc")
        pair_sur.append([p, r["scheme"], fmt(r), fmt(best_row(df, pair=p, task="Survival", metric="c_index"))])
    add_table(doc, ["组合", "最优方案", "ROC AUC", "C-index"], pair_sur, widths=[1.3, 1.7, 1.7, 1.8])
    doc.add_picture(str(FIG / "fig_pairwise_integration.png"), width=Inches(6.2))
    add_caption(doc, "图 3. 两两组学整合最优结果")
    add_description(doc, "图 3 左图展示 PAM50 两两组合最优 Accuracy，横轴为三种组合，蓝色柱为对应组合的最优方案 Accuracy，误差线表示标准差。mRNA+CNV 最高，mRNA+miRNA 次之，CNV+miRNA 最低。右图展示生存预测最优 ROC AUC，CNV+miRNA 和 mRNA+CNV 较高，mRNA+miRNA 最低。")

    doc.add_heading("5. 9 模型 Stacking 与两两 Stacking", level=1)
    df = read("final_9model_stacking_results.tsv")
    stack = df[df.model == "Stacking_9"]
    add_table(doc, ["任务", "指标", "结果"], [
        ["PAM50", "Accuracy", fmt(stack[(stack.task == "PAM50") & (stack.metric == "accuracy")].iloc[0])],
        ["PAM50", "Macro-F1", fmt(stack[(stack.task == "PAM50") & (stack.metric == "macro_f1")].iloc[0])],
        ["Survival", "ROC AUC", fmt(stack[(stack.task == "Survival") & (stack.metric == "roc_auc")].iloc[0])],
        ["Survival", "C-index", fmt(stack[(stack.task == "Survival") & (stack.metric == "c_index")].iloc[0])],
    ], widths=[1.2, 1.4, 3.4])
    add_description(doc, "9 模型 Stacking 的 PAM50 Accuracy 为 0.9229，低于最优单组学；Survival ROC AUC 为 0.7469，高于单模型，但 C-index 低于三组学 Concat+MLP。")

    df2 = read("pairwise_stacking_results.tsv")
    pair_stack = df2.to_dict("records")
    add_table(doc, ["组合", "任务", "指标", "结果"], [
        [r["pair"], r["task"], r["metric"], fmt(r)] for r in pair_stack
    ], widths=[1.2, 1.1, 1.2, 2.5])

    doc.add_heading("6. 更复杂整合方法", level=1)
    advanced = [
        ["Multi-task PAM50+Survival", "PAM50 Accuracy", "0.8683"],
        ["Multi-task PAM50+Survival", "Survival C-index", "0.6519"],
        ["DeepSurv", "Survival C-index", "0.6534"],
        ["Transformer", "PAM50 Accuracy", "0.9270"],
        ["Transformer", "PAM50 Macro-F1", "0.9288"],
        ["Transformer", "Survival C-index", "0.6938"],
        ["Low-rank bilinear", "PAM50 Accuracy", "0.7221"],
        ["GNN", "PAM50 Accuracy", "0.8702"],
        ["GNN", "Survival C-index", "0.6493"],
    ]
    add_table(doc, ["方法", "任务/指标", "结果"], advanced, widths=[2.2, 2.0, 2.3])
    doc.add_picture(str(FIG / "fig_advanced_stacking.png"), width=Inches(6.2))
    add_caption(doc, "图 4. 更复杂整合方法总结")
    add_description(doc, "图 4 左图以水平条形图比较 PAM50 更复杂整合方法的 Accuracy，横轴为 Accuracy，误差线为 5 折标准差。Transformer 与单组学 mRNA ML 最高，9 模型 Stacking 略低，GNN 和低秩双线性融合较低。右图比较 Survival C-index，三组学 Concat MLP 和 CNN 晚期平均最高，9 模型 Stacking 略低，DeepSurv 和 Transformer 更低。")

    doc.add_heading("7. 配对检验", level=1)
    df = read("comprehensive_paired_tests_selected.tsv")
    add_table(doc, ["任务", "指标", "比较", "均值差", "t p", "Wilcoxon p"], [
        [r["task"], r["metric"], r["comparison"], r["mean_diff"], r["t_pvalue"], r["wilcoxon_pvalue"]] for r in df.to_dict("records")
    ], widths=[0.9, 1.0, 2.0, 0.9, 0.8, 0.9])
    add_description(doc, "在 5 折样本下，最优全面融合相对最优单组学和最优组学内算法融合均未达到显著差异。")

    doc.add_heading("8. 讨论", level=1)
    discussion = [
        "三组学整合和两两组学整合的结果表明，融合策略的效果高度依赖任务类型。PAM50 分型中，mRNA 单组学已经提供较强的线性判别信息，简单特征拼接或 Transformer 可以在 Macro-F1 上略有改善，但整体提升有限。",
        "生存预测中，三组学特征拼接 MLP 和 CNN 晚期平均融合明显优于任一组学单组学，说明生存风险信号更依赖多组学的非线性组合。",
        "9 模型 Stacking 虽然引入了更复杂的 meta-model，但在当前样本量下没有稳定优于简单拼接和晚期平均。其原因是基础模型之间存在高度相关性，Stacking 无法充分增加多样性，同时内层交叉验证进一步减少了可用训练样本。",
        "DeepSurv、低秩双线性融合和简化图神经网络等复杂方法在理论上有更强的表达能力，但当前实现和样本规模下未超过基线。这提示在 TCGA 小样本多组学图像化研究中，模型复杂度和可用样本量之间的平衡比单纯增加网络结构更重要。",
        "配对检验未发现最优融合相对最优单组学存在显著优势，因此论文中应谨慎表述整合效果，建议使用“与单组学相当、无显著劣化”，而不是宣称显著提升。",
    ]
    for text in discussion:
        p = doc.add_paragraph(); p.add_run(text)

    doc.add_heading("9. 结论", level=1)
    for c in [
        "PAM50 分型推荐使用 mRNA 单组学 LogisticRegression 或 Transformer 跨组学注意力；多组学整合未形成显著增益。",
        "Survival 生存预测推荐使用三组学特征拼接 MLP 或 CNN 晚期平均融合。",
        "两两组学整合中，mRNA+CNV 在 PAM50 上最强；CNV+miRNA 的 CNN 晚期平均在生存预测中具有较高 ROC AUC。",
        "9 模型 Stacking 和两两 Stacking 未稳定超过简单拼接或晚期融合。",
        "DeepSurv、低秩双线性融合和 GNN 当前未超过基线，后续需进一步调参或引入更丰富的生物学先验。",
    ]:
        p = doc.add_paragraph(style="List Bullet"); p.add_run(c)

    doc.save(OUT)
    print("Saved ->", OUT)


if __name__ == "__main__":
    main()
