#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build academic-paper-style PAM50 feature-selection manuscript."""

from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "data" / "figures"
OUT = ROOT / "data" / "PAM50特征提取筛选_学术论文版_v2.docx"
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
    r.font.name="Calibri"; r.font.size=Pt(16 if level==1 else (13 if level==2 else 12)); r.font.bold=True; r.font.color.rgb=BLUE
    h.paragraph_format.space_before=Pt(12); h.paragraph_format.space_after=Pt(6)


def add_body(doc, text):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    r=p.add_run(text); r.font.name="Calibri"; r.font.size=Pt(11)
    p.paragraph_format.space_after=Pt(6); p.paragraph_format.line_spacing=1.25


def add_equation(doc, omml):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p._p.append(parse_xml(omml))
    p.paragraph_format.space_after = Pt(6)


def add_figure(doc, path, caption):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; run=p.add_run(); run.add_picture(str(path), width=Inches(6.0))
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


def main():
    doc=Document(); sec=doc.sections[0]
    sec.page_width=Inches(8.5); sec.page_height=Inches(11)
    sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Inches(1)
    normal=doc.styles["Normal"]; normal.font.name="Calibri"; normal.font.size=Pt(11)

    title=doc.add_paragraph(); title.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=title.add_run("基于多组学特征筛选与机器学习的乳腺癌 PAM50 分子分型研究"); r.font.name="Calibri"; r.font.size=Pt(17); r.font.bold=True

    add_heading(doc,"摘要",1)
    add_body(doc,"乳腺癌具有高度分子异质性，PAM50 分子分型对精准治疗具有重要意义。本研究系统比较了低方差过滤、F 值、L1、NSRE 及其两两组合等特征提取方法，并评估逻辑回归、随机森林、梯度提升、支持向量机、K 近邻和多层感知器等模型在不同特征数量下的分类性能。结果表明，mRNA 使用 V_L1 特征方法结合 MLP、保留 400 个特征时，PAM50 四分类准确率达到 0.9136；miRNA 使用 V_F_NSRE 结合 MLP、600 个特征时准确率达到 0.8471；CNV 判别能力较弱，50 个特征即可。该研究为乳腺癌多组学特征筛选和模型选择提供了系统依据。")

    add_heading(doc,"1. 引言",1)
    add_body(doc,"乳腺癌是全球女性最常见的恶性肿瘤之一，其分子分型对治疗方案选择和预后判断具有决定性作用。PAM50 分型将乳腺癌分为 Luminal A、Luminal B、HER2-enriched、Basal-like 和 Normal-like 等亚型。随着高通量测序技术的发展，mRNA、CNV、miRNA 等多组学数据为分子分型提供了丰富信息，但高维、小样本、多源异质性等问题给特征筛选和模型构建带来挑战。传统机器学习方法通常将组学特征以向量形式输入，难以充分利用特征间的结构关系。本研究从特征提取算法、特征数量和机器学习模型三个维度系统评估 PAM50 分型性能，为多组学融合分析提供基础。")

    add_heading(doc,"2. 材料与方法",1)
    add_heading(doc,"2.1 数据来源",2)
    add_body(doc,"数据来源于 TCGA-BRCA 项目，通过 UCSC Xena 获取 mRNA（HiSeqV2）、CNV（GISTIC2 阈值化）和 miRNA（miRNA_HiSeq_gene）表达矩阵，以及 PAM50 分型标签。样本纳入标准为具有对应组学数据且 PAM50 标签非缺失。mRNA 可用 833 例，CNV 818 例，miRNA 497 例，三组学交集 493 例。")
    add_table(doc, ["组学","特征方法","特征数","样本数"], [
        ["mRNA","V_L1","400","833"],
        ["CNV","V_L1_F","50","818"],
        ["miRNA","V_F_NSRE","600","497"],
    ], [1600,2200,1600,1600])

    add_heading(doc,"2.2 数据预处理",2)
    add_body(doc,"对每组学分别进行样本对齐，优先选择原发肿瘤样本。建模前在交叉验证训练折内部进行标准化，避免信息泄漏。")

    add_heading(doc,"2.3 特征提取算法",2)
    add_heading(doc,"2.3.1 低方差过滤",3)
    add_body(doc,"低方差过滤用于删除在不同样本之间几乎没有变化的特征。对于一个特征 x_j，其方差定义为 Var(x_j)=Σ_i (x_ij - x̄_j)^2 / (n-1)。当 Var(x_j)<τ 时，该特征被认为缺乏判别信息并被删除。本研究设置 τ=0，即删除所有方差为零的特征。该步骤可以降低数据维度，减少后续模型计算量。")
    add_equation(doc, '<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath><m:f><m:num><m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="subSup"/></m:naryPr><m:sub><m:r><m:t>i=1</m:t></m:r></m:sub><m:sup><m:r><m:t>n</m:t></m:r></m:sup><m:e><m:sSup><m:e><m:r><m:t>(x</m:t></m:r></m:e><m:sup><m:r><m:t>ij</m:t></m:r></m:sup></m:sSup><m:r><m:t> - x̄</m:t></m:r><m:sSub><m:e><m:r><m:t>j</m:t></m:r></m:e><m:sub><m:r><m:t>j</m:t></m:r></m:sub></m:sSub><m:r><m:t>)^2</m:t></m:r></m:e></m:nary></m:num><m:den><m:r><m:t>n-1</m:t></m:r></m:den></m:f></m:oMath></m:oMathPara>')
    add_heading(doc,"2.3.2 F 值（ANOVA）",3)
    add_body(doc,"F 值通过单因素方差分析衡量一个特征在多个类别之间的均值差异是否显著。F=MS_between/MS_within，其中 MS_between 为组间均方，MS_within 为组内均方。F 值越大，说明该特征在不同类别间的差异越大，判别能力越强。本研究使用 F 值对所有特征进行排序，并选择 F 值最大的前 k 个特征。")
    add_equation(doc, '<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath><m:f><m:num><m:r><m:t>MS_between</m:t></m:r></m:num><m:den><m:r><m:t>MS_within</m:t></m:r></m:den></m:f></m:oMath></m:oMathPara>')
    add_heading(doc,"2.3.3 L1/LASSO",3)
    add_body(doc,"L1 正则化通过在损失函数中加入权重的绝对值惩罚，使部分权重收缩为零，从而实现特征选择。其优化目标为 min_w (1/2)||Xw-y||^2 + λ||w||_1。λ 控制稀疏程度，λ 越大，被保留的特征越少。本研究使用带 L1 惩罚的线性 SVM 进行特征选择，通过调整惩罚参数得到指定数量的特征。")
    add_equation(doc, '<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath><m:r><m:t>min_w</m:t></m:r><m:r><m:t>  (1/2)||Xw-y||</m:t></m:r><m:sSup><m:e><m:r><m:t>2</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup><m:r><m:t> + λ||w||</m:t></m:r><m:sSub><m:e><m:r><m:t>1</m:t></m:r></m:e><m:sub><m:r><m:t>1</m:t></m:r></m:sub></m:sSub></m:oMath></m:oMathPara>')
    add_heading(doc,"2.3.4 NSRE 新对称相对熵",3)
    add_body(doc,"NSRE 通过比较正负类样本特征值概率分布的差异来评价特征重要性。对每个特征，先按分位数将连续值离散为若干区间，分别估计正类和负类样本的分布 p 和 q，然后计算 NSRE(p||q)=Σ p_i log2(2p_i/(p_i+q_i)) + Σ q_i log2(2q_i/(p_i+q_i))。该值越大，说明两类分布差异越大，特征越具有判别性。")
    add_equation(doc, '<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath><m:r><m:t>NSRE(p||q)=</m:t></m:r><m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="subSup"/></m:naryPr><m:sub><m:r><m:t>i</m:t></m:r></m:sub><m:sup><m:r><m:t> </m:t></m:r></m:sup><m:e><m:r><m:t>p_i log2(2p_i/(p_i+q_i)) + q_i log2(2q_i/(p_i+q_i))</m:t></m:r></m:e></m:nary></m:oMath></m:oMathPara>')
    add_heading(doc,"2.3.5 组合方法",3)
    add_body(doc,"为兼顾不同筛选策略，本研究将上述方法进行两两组合，形成九种特征提取流程：V_F、V_L1、V_NSRE、V_F_L1、V_F_NSRE、V_L1_F、V_L1_NSRE、V_NSRE_F、V_NSRE_L1。其中 V 表示先做低方差过滤，符号“_”后的方法依次执行。组合方式可以先用 F 值或 L1 进行粗筛，再用 NSRE 进行精筛，从而在保留判别特征的同时降低计算复杂度。")
    table = doc.add_table(rows=1, cols=3)
    hdr = table.rows[0].cells
    for i, txt in enumerate(["组合方法", "执行流程", "说明"]):
        hdr[i].text = txt; set_cell_shading(hdr[i], "F2F4F7")
    rows = [
        ("V_F", "低方差过滤 → F 值", "先删除低方差特征，再按 F 值选择 top-k"),
        ("V_L1", "低方差过滤 → L1", "先删除低方差特征，再用 L1 稀疏选择"),
        ("V_NSRE", "低方差过滤 → NSRE", "先删除低方差特征，再按 NSRE 选择"),
        ("V_F_L1", "低方差过滤 → F 值 → L1", "F 值粗筛，L1 精筛"),
        ("V_F_NSRE", "低方差过滤 → F 值 → NSRE", "F 值粗筛，NSRE 精筛"),
        ("V_L1_F", "低方差过滤 → L1 → F 值", "L1 粗筛，F 值精筛"),
        ("V_L1_NSRE", "低方差过滤 → L1 → NSRE", "L1 粗筛，NSRE 精筛"),
        ("V_NSRE_F", "低方差过滤 → NSRE → F 值", "NSRE 粗筛，F 值精筛"),
        ("V_NSRE_L1", "低方差过滤 → NSRE → L1", "NSRE 粗筛，L1 精筛"),
    ]
    for row in rows:
        cells = table.add_row().cells
        for i, txt in enumerate(row):
            cells[i].text = txt
    set_table_widths(table, [1800, 3600, 3960])

    add_heading(doc,"2.4 机器学习模型",2)
    add_heading(doc,"2.4.1 LogisticRegression",3)
    add_body(doc,"逻辑回归通过 sigmoid 函数将线性组合映射到 0-1 之间的概率，适用于二分类和多分类。模型使用 L2 正则化，最大迭代次数设为 2000。")
    add_equation(doc, '<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath><m:r><m:t>P(y=1|x)=1/(1+exp(-(w^T x+b)))</m:t></m:r></m:oMath></m:oMathPara>')
    add_heading(doc,"2.4.2 RandomForest",3)
    add_body(doc,"随机森林通过自助采样构建多棵决策树，并在节点分裂时随机选择特征子集，最终通过多数投票得到分类结果。本研究设置 n_estimators=100。")
    add_heading(doc,"2.4.3 GradientBoosting",3)
    add_body(doc,"梯度提升树通过逐步拟合前一轮模型的残差来构建加性模型，使用梯度下降思想优化损失函数。本研究设置 n_estimators=100。")
    add_heading(doc,"2.4.4 SVC",3)
    add_body(doc,"支持向量机通过寻找最大间隔超平面进行分类。本研究采用线性核 SVC。")
    add_equation(doc, '<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath><m:r><m:t>min_w (1/2)||w||^2 + C Σ max(0, 1-y_i(w^T x_i+b))</m:t></m:r></m:oMath></m:oMathPara>')
    add_heading(doc,"2.4.5 KNN",3)
    add_body(doc,"K 近邻根据测试样本与训练样本的距离，选择最近的 k 个邻居，通过多数投票确定类别。本研究设置 k=5。")
    add_heading(doc,"2.4.6 MLP",3)
    add_body(doc,"多层感知器由输入层、隐藏层和输出层组成，通过非线性激活函数和反向传播学习特征表示。本研究结构为输入-128-ReLU-Dropout(0.3)-64-ReLU-Dropout(0.3)-输出，使用 Adam 优化器，学习率 0.001，权重衰减 0.0001，训练 15 个 epoch。")
    add_equation(doc, '<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath><m:r><m:t>h=ReLU(W x + b)</m:t></m:r></m:oMath></m:oMathPara>')

    add_heading(doc,"2.5 实验设计",2)
    add_body(doc,"mRNA、CNV、miRNA 基础特征数分别为 200、50、200，并按 2、3、4、5 倍扩增。所有组合采用 5 折分层交叉验证，随机种子 42，记录 Accuracy 和 Macro-F1 的均值与标准差。")
    table = doc.add_table(rows=1, cols=3)
    hdr = table.rows[0].cells
    for i, txt in enumerate(["项目", "设置", "说明"]):
        hdr[i].text = txt; set_cell_shading(hdr[i], "F2F4F7")
    rows = [
        ("组学", "mRNA / CNV / miRNA", "三组学分别独立分析"),
        ("基础特征数", "200 / 50 / 200", "对应 mRNA / CNV / miRNA"),
        ("特征倍数", "1× / 2× / 3× / 4× / 5×", "扩增保留特征数量"),
        ("特征方法", "10 种", "V 及九种两两组合"),
        ("机器学习模型", "LR / RF / GB / SVC / KNN / MLP", "传统 ML 与深度学习"),
        ("交叉验证", "5 折分层交叉验证", "random_state=42"),
        ("评价指标", "Accuracy / Macro-F1", "记录均值与标准差"),
    ]
    for row in rows:
        cells = table.add_row().cells
        for i, txt in enumerate(row):
            cells[i].text = txt
    set_table_widths(table, [1600, 3600, 4160])
    add_body(doc,"综上，本研究共涉及 10 种特征方法 × 6 种机器学习模型 × 3 种组学 × 5 个特征倍数 = 900 组实验组合；每组均进行 5 折分层交叉验证，并记录 Accuracy 与 Macro-F1 的均值和标准差。")

    add_heading(doc,"3. 结果",1)
    add_body(doc,"本研究在 PAM50 四分类任务上系统比较了 10 种特征提取方法、5 个特征倍数和 6 种机器学习模型的性能，共 900 组实验组合，每组均采用 5 折分层交叉验证，并以 Accuracy 和 Macro-F1 的均值与标准差作为评价指标。")
    add_body(doc,"总体来看，MLP 在 mRNA、CNV、miRNA 三组学上均优于 LogisticRegression、RandomForest、GradientBoosting、SVC 和 KNN；其中 KNN 在高维特征下表现最不稳定。三组学的判别能力排序为：mRNA 最强，miRNA 次之，CNV 最弱。")
    add_body(doc,"在特征数量方面，mRNA 使用 V_L1 方法、保留 400 个特征时 Accuracy 为 0.9136，继续增加到 1000 特征仅带来约 0.0036 的微弱提升；miRNA 使用 V_F_NSRE 方法、保留 600 个特征时达到 0.8471 的性能拐点；CNV 使用 V_L1_F 方法、50 个特征时 Accuracy 为 0.7078，扩增特征数没有稳定收益。")
    add_body(doc,"以下图 1 至图 3 分别从特征数量、模型类型和所有算法三个角度展示结果，并通过误差棒反映 5 折交叉验证的波动。")
    add_body(doc,"关于模型间的差异显著性：本研究当前仅保存了各模型的 5 折均值和标准差，未保存每一折的预测分数，因此无法直接进行配对 t 检验或 Wilcoxon 符号秩检验。从标准差来看，多数模型之间的 Accuracy 差异在 0.02–0.05 左右，部分模型差异小于 1 个标准差，提示差异可能不具有统计显著性；而 MLP 与 KNN 之间的差距通常超过 1 个标准差，可能具有统计学意义。后续应在交叉验证过程中保存逐折分数，以便正式检验模型间的显著性。")
    add_figure(doc, FIG/"pam50_best_methods_by_k.png", "图 1. 各组学最优特征方法在不同特征数下的 Accuracy（mean ± std）")
    add_figure(doc, FIG/"pam50_best_methods_by_k_macro_f1.png", "图 1b. 各组学最优特征方法在不同特征数下的 Macro-F1（mean ± std）")
    add_figure(doc, FIG/"pam50_lr_vs_mlp.png", "图 2. LogisticRegression 与 MLP 最优 Accuracy 对比")
    add_figure(doc, FIG/"pam50_lr_vs_mlp_macro_f1.png", "图 2b. LogisticRegression 与 MLP 最优 Macro-F1 对比")
    add_figure(doc, FIG/"pam50_all_models.png", "图 3. 各组学所有机器学习算法最优 Accuracy 对比（mean ± std）")
    add_figure(doc, FIG/"pam50_all_models_macro_f1.png", "图 3b. 各组学所有机器学习算法最优 Macro-F1 对比（mean ± std）")
    add_body(doc,"图 1 和图 1b 分别展示了各组学最优特征方法在不同特征数量下的 Accuracy 与 Macro-F1。mRNA 的 V_L1+MLP 在 400 特征时即接近饱和，miRNA 的 V_F_NSRE+MLP 在 600 特征时达到拐点，CNV 在 50 特征时已足够。")
    add_body(doc,"图 2 和图 2b 比较了 LogisticRegression 与 MLP，结果表明 MLP 在各组学上均优于逻辑回归。图 3 和图 3b 进一步展示了所有机器学习算法的最优 Accuracy 与 Macro-F1，其中 MLP 表现最好，KNN 在多数组学上较弱。")
    add_body(doc,"综合结果见表 1。")
    add_heading(doc,"表 1 各组学最优组合",2)
    table=doc.add_table(rows=1, cols=6)
    hdr=table.rows[0].cells
    for i,txt in enumerate(["组学","特征数","倍数","特征方法","模型","Accuracy"]):
        hdr[i].text=txt; set_cell_shading(hdr[i],"F2F4F7")
    for row in [("mRNA","400","2×","V_L1","MLP","0.9136"),("miRNA","600","3×","V_F_NSRE","MLP","0.8471"),("CNV","50","1×","V_L1_F","MLP","0.7078")]:
        cells=table.add_row().cells
        for i,txt in enumerate(row): cells[i].text=txt
    set_table_widths(table,[1000,1000,1000,1800,1200,1200])

    add_heading(doc,"表 1 各组学最优组合的理由",2)
    add_body(doc,"mRNA：V_L1 + MLP + 400 特征的 Accuracy 为 0.9136。虽然 1000 特征时 Accuracy 略高（0.9172），但提升仅约 0.0036，计算成本却大幅增加，因此 400 特征在精度和效率之间更优。")
    add_body(doc,"miRNA：V_F_NSRE + MLP + 600 特征的 Accuracy 为 0.8471。当特征数从 200 增加到 600 时性能明显提升，但继续增加到 800 或 1000 后性能下降，因此 600 是性能拐点。")
    add_body(doc,"CNV：V_L1_F + MLP + 50 特征的 Accuracy 为 0.7078。CNV 本身判别能力较弱，增加特征数没有带来稳定提升，50 个特征已经足够，且计算效率最高。")

    add_heading(doc,"4. 讨论",1)
    add_body(doc,"mRNA 在 V_L1+MLP、400 特征时即可达到约 0.914 的 Accuracy，继续增加到 1000 特征收益很小，因此 400 特征具有更好的计算效率。miRNA 在 600 特征时达到性能拐点。CNV 整体判别能力较弱，保持 50 特征即可。MLP 在各组学上均优于传统机器学习模型，表明非线性模型更适合高维多组学特征。")

    add_heading(doc,"5. 结论",1)
    add_body(doc,"本研究系统比较了多种特征提取方法和机器学习模型在乳腺癌 PAM50 分型中的表现，给出了兼顾精度与计算效率的特征数量组合。未来将在此基础上进行多组学融合和图像化建模。")

    doc.save(OUT); print(f"Saved -> {OUT}")


if __name__=="__main__":
    main()
