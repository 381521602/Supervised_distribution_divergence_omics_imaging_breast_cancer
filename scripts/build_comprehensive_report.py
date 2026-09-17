#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build comprehensive final report."""

from pathlib import Path
import pandas as pd
from build_final_results_tables_docx import add_table_from_df
from build_integration_tables_docx import add_table as add_integration_table
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "data" / "figures"
OUT = ROOT / "data" / "多组学建模综合报告.docx"
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


def add_figure(doc,path,caption):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; run=p.add_run(); run.add_picture(str(path),width=Inches(6.0))
    c=doc.add_paragraph(); c.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=c.add_run(caption); r.font.name="Calibri"; r.font.size=Pt(9); r.font.color.rgb=RGBColor(0x55,0x55,0x55)


def add_table(doc, headers, rows, widths):
    table=doc.add_table(rows=1, cols=len(headers))
    hdr=table.rows[0].cells
    for i,txt in enumerate(headers):
        hdr[i].text=txt; set_cell_shading(hdr[i],"F2F4F7")
    for row in rows:
        cells=table.add_row().cells
        for i,txt in enumerate(row): cells[i].text=txt
    set_table_widths(table, widths)
    return table


def add_full_results_table(doc, df, headers, metrics):
    table=doc.add_table(rows=1, cols=len(headers))
    hdr=table.rows[0].cells
    for i,txt in enumerate(headers):
        hdr[i].text=txt; set_cell_shading(hdr[i],"F2F4F7")
    group_col = "omics" if "omics" in df.columns else "combo"
    for group in sorted(df[group_col].unique()):
        for model in sorted(df["model"].unique()):
            sub=df[(df[group_col]==group)&(df.model==model)]
            if sub.empty:
                continue
        cells=table.add_row().cells
        cells[0].text=group
        cells[1].text=model
        for j,metric in enumerate(metrics, start=2):
            m=sub[sub.metric==metric]
            if len(m)>0:
                r=m.iloc[0]
                cells[j].text=f"{r['mean']:.4f} ± {r['std']:.4f}"
    set_table_widths(table, [1200]+[2200 for _ in range(len(headers)-1)])


def main():
    single=pd.read_csv(ROOT/"data/final_datasets_ml_results.tsv",sep="\t")
    integ=pd.read_csv(ROOT/"data/integration_final_datasets_results.tsv",sep="\t")
    integ_mlp=pd.read_csv(ROOT/"data/integration_mlp_cox_results.tsv",sep="\t")
    doc=Document(); sec=doc.sections[0]
    sec.page_width=Inches(8.5); sec.page_height=Inches(11)
    sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Inches(1)
    normal=doc.styles["Normal"]; normal.font.name="Calibri"; normal.font.size=Pt(11)

    title=doc.add_paragraph(); title.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=title.add_run("乳腺癌多组学数据集构建、单组学预测与多组学整合研究报告"); r.font.name="Calibri"; r.font.size=Pt(17); r.font.bold=True
    add_body(doc,"注：本文所有模型结果均为基于预选特征的结果，即特征选择在交叉验证前完成，模型训练仍采用 5 折交叉验证。")

    add_heading(doc,"1. 数据集构建",1)
    add_body(doc,"基于 TCGA-BRCA 的 mRNA、CNV、miRNA 数据，采用最优特征方法提取特征，并保留特征名用于可解释性分析。PAM50 分型数据集：mRNA 使用 V_L1 保留 400 特征，CNV 使用 V_L1_F 保留 50 特征，miRNA 使用 V_F_JSD 保留 600 特征。生存预测数据集：mRNA 使用 V_JSD_F 保留 200 特征，CNV 使用 V_L1_JSD 保留 150 特征，miRNA 使用 V_L1_JSD 保留 600 特征。所有数据集均记录样本 ID、标签和特征名。")
    add_table(doc, ["任务","组学","特征方法","特征数","样本数"], [
        ["PAM50","mRNA","V_L1","400","833"],
        ["PAM50","CNV","V_L1_F","50","818"],
        ["PAM50","miRNA","V_F_JSD","600","497"],
        ["生存预测","mRNA","V_JSD_F","200","1073"],
        ["生存预测","CNV","V_L1_JSD","150","1057"],
        ["生存预测","miRNA","V_L1_JSD","600","739"],
    ], [1600,1200,2200,1200,1200])

    add_heading(doc,"2. 单组学预测",1)
    add_heading(doc,"2.1 PAM50 分型",2)
    add_body(doc,"基于预选特征，5 折交叉验证。mRNA 最优模型为 LogisticRegression，Accuracy 0.9628；CNV 最优为 MLP，Accuracy 0.7237；miRNA 最优为 MLP，Accuracy 0.8451。")
    add_body(doc,"从表 2 可见，mRNA 的 LogisticRegression 与 SVC 均达到 0.95 以上，说明 mRNA 特征包含较强的 PAM50 判别信息；RandomForest 和 GradientBoosting 略低，KNN 明显较差。CNV 各模型 Accuracy 集中在 0.67–0.72，Macro-F1 较低，说明 CNV 单独判别能力有限。miRNA 中 MLP 与 LogisticRegression 接近 0.84，优于树模型和 KNN。")
    add_table_from_df(doc, single[single.task=="PAM50"], "omics", ["LogisticRegression","RandomForest","GradientBoosting","SVC","KNN","MLP"], ["accuracy","macro_f1"])
    add_figure(doc, FIG/"pam50_all_models.png", "图 1. PAM50 分型各组学所有机器学习算法最优 Accuracy")
    add_figure(doc, FIG/"pam50_all_models_macro_f1.png", "图 2. PAM50 分型各组学所有机器学习算法最优 Macro-F1")

    add_heading(doc,"2.2 生存预测",2)
    add_body(doc,"生存预测中，CNV 表现最好。CoxPH 的 C-index 达到 0.7297，LogisticRegression 的 ROC AUC 达到 0.6977。")
    add_body(doc,"从表 3 可见，CNV 在 ROC AUC 和 C-index 上均优于 mRNA 和 miRNA，说明 CNV 对生存结局具有更强的判别信息。CoxPH 在各组学上的 C-index 普遍高于传统分类器基于概率计算得到的 C-index，表明时间到事件模型更适合生存数据。")
    add_table_from_df(doc, single[single.task=="Survival"], "omics", ["LogisticRegression","RandomForest","GradientBoosting","SVC","KNN","CoxPH"], ["roc_auc","c_index"])
    add_table(doc, ["组学","模型","ROC AUC","C-index"], [
        ["mRNA","LogisticRegression","0.6192 ± 0.0270","0.6390 ± 0.0681"],
        ["mRNA","CoxPH","0.6701 ± 0.0756","0.7178 ± 0.0472"],
        ["CNV","LogisticRegression","0.6977 ± 0.0328","0.7007 ± 0.0261"],
        ["CNV","CoxPH","0.7194 ± 0.0401","0.7297 ± 0.0363"],
        ["miRNA","LogisticRegression","0.6126 ± 0.0508","0.5665 ± 0.0521"],
        ["miRNA","CoxPH","0.5507 ± 0.0597","0.5599 ± 0.0472"],
    ], [1200,2200,2400,2400])
    add_figure(doc, FIG/"survival_best_methods_cox.png", "图 3. 生存预测 Cox C-index 最优特征方法")
    add_figure(doc, FIG/"survival_best_methods_auc.png", "图 4. 生存预测 AUC 最优特征方法")

    add_heading(doc,"3. 多组学整合",1)
    add_heading(doc,"3.1 传统机器学习整合",2)
    add_body(doc,"PAM50 分型中，mRNA+CNV 使用 LogisticRegression 达到 Accuracy 0.9486；生存预测中，mRNA+CNV 的 RandomForest C-index 为 0.6308。")
    add_body(doc,"从整合结果可见，PAM50 分型中 mRNA+CNV 是最优两两组合，三组学拼接与 mRNA+miRNA 接近，而 CNV+miRNA 明显较差。生存预测中 mRNA+CNV 的 C-index 最高，但未超过 CNV 单组学 CoxPH，说明简单拼接会稀释 CNV 的生存信号。")
    add_integration_table(doc, integ[integ.task=="PAM50"], "combo", ["LogisticRegression","RandomForest","GradientBoosting","SVC","KNN"], ["accuracy","macro_f1"])
    add_integration_table(doc, integ[integ.task=="Survival"], "combo", ["LogisticRegression","RandomForest","GradientBoosting","SVC","KNN"], ["roc_auc","c_index"])
    add_table(doc, ["组合","PAM50 最优 Accuracy","生存最优 C-index"], [
        ["mRNA+CNV","0.9486","0.6308"],
        ["mRNA+miRNA","0.9175","0.5813"],
        ["CNV+miRNA","0.6639","0.5861"],
        ["三组学","0.9075","0.5792"],
    ], [2200,2600,2600])
    add_heading(doc,"3.2 MLP 与 CoxPH 整合",2)
    add_body(doc,"PAM50 分型中，mRNA+CNV 使用 MLP 达到 Accuracy 0.9523；生存预测中，mRNA+CNV 使用 CoxPH 的 C-index 为 0.7013。")
    add_body(doc,"MLP 整合在 mRNA+CNV 上略优于传统 LogisticRegression，但在三组学拼接中未超过 mRNA 单组学。CoxPH 整合在 mRNA+CNV 上 C-index 为 0.7013，仍低于 CNV 单组学 CoxPH 的 0.7297。")

    add_heading(doc,"讨论",1)
    add_body(doc,"本研究表明，mRNA 是 PAM50 分型的最主要组学，CNV 是生存预测的最主要组学。简单特征拼接融合没有带来稳定提升，主要原因是不同组学的判别信号和尺度差异较大，等权拼接会引入冗余和噪声。后续应重点探索加权晚期融合、注意力机制或多视图学习，使模型能够自适应地学习各组学的重要性。")
    add_body(doc,"关于模型差异显著性，当前基于预选特征的结果存在轻微信息泄漏，且未保存逐折预测分数，因此不能直接进行配对统计检验。论文中应注明基于预选特征的结果，并在后续工作中补充严格嵌套交叉验证。")
    add_integration_table(doc, integ_mlp[integ_mlp.task=="PAM50"], "combo", ["MLP"], ["accuracy","macro_f1"])
    add_integration_table(doc, integ_mlp[integ_mlp.task=="Survival"], "combo", ["CoxPH"], ["roc_auc","c_index"])
    add_table(doc, ["组合","MLP Accuracy","CoxPH C-index"], [
        ["mRNA+CNV","0.9523","0.7013"],
        ["mRNA+miRNA","0.9035","0.6151"],
        ["CNV+miRNA","0.6538","0.6141"],
        ["三组学","0.9015","0.5948"],
    ], [2200,2600,2600])

    add_heading(doc,"5. 计算过程说明",1)
    add_body(doc,"ROC AUC：将模型输出的死亡风险概率作为预测分数，使用 sklearn.metrics.roc_auc_score 计算，衡量二分类判别能力。")
    add_body(doc,"C-index：将同一风险分数（概率或决策函数值）与生存时间、事件状态结合，使用 lifelines.utils.concordance_index 计算；分数越大表示风险越高，C-index 衡量预测排序与真实生存顺序的一致性。")
    add_body(doc,"CoxPH 直接输出部分风险，再计算 C-index；其他模型没有 Cox 风险函数，所以用其预测概率作为风险分数。CoxPH 本身不直接输出事件发生概率，但可以将其部分风险分数作为连续预测值，结合死亡事件标签用 roc_auc_score 计算 ROC AUC；该 AUC 表示风险分数对死亡事件的判别能力，不等同于 Cox 模型的时间预测性能。")

    add_heading(doc,"4. 结论",1)
    add_body(doc,"简单特征拼接融合未超过最优单组学结果。PAM50 分型以 mRNA 单组学最优；生存预测以 CNV 单组学最优。后续应尝试加权晚期融合或注意力机制。")

    doc.save(OUT); print(f"Saved -> {OUT}")


if __name__=="__main__":
    main()
