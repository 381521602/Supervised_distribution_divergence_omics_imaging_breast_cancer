# Data and Code Manifest

## Purpose

This document describes each major component in the repository and how the files map to the manuscript analyses.

## 1. Clinical and molecular labels

| File | Description |
| --- | --- |
| `data/brca_labels_modeling_ready.tsv` | Case-level labels used for PAM50 four-class and overall survival modeling. |
| `data/brca_molecular_subtype.tsv` | Molecular subtype annotation extracted for exploratory checks. |
| `data/brca_clinical.tsv` | TCGA-BRCA clinical fields used for survival status and time. |
| `data/brca_unified_manifest.tsv` | Unified sample manifest for alignment checks. |
| `data/brca_files.tsv` | Source-file inventory and sample barcode mapping. |
| `data/sample_level_provenance.tsv` | Case-level provenance: sample IDs, sample type, PAM50 label, OS time/event, and omics availability. |
| `data/omics_functional_category_annotation.tsv` | Complete feature-to-functional-category mapping used in the masking and interpretability analyses. |

## 2. Final pre-selected feature matrices

| Path | Task |
| --- | --- |
| `data/final_datasets/PAM50/*.tsv` | Final mRNA/CNV/miRNA feature matrices for PAM50 four-class classification. |
| `data/final_datasets/Survival/*.tsv` | Final mRNA/CNV/miRNA feature matrices for overall survival prediction. |
| `data/selected_features/*` | Feature names, selected matrices, and JSD score files. |

## 3. Omics images

| Path | Description |
| --- | --- |
| `data/images/PAM50/` | JSD-ordered images for PAM50 single-omics CNN models. |
| `data/images/Survival/` | JSD-ordered images for survival single-omics CNN models. |
| `data/images/examples/` | Example grayscale, JSD, and functional-category grids. |
| `data/images/report_figures/` | FullSizeCNN architecture and report-level comparison figures. |
| `data/paper_figures/` | Final manuscript figures referenced by `build_manuscript_and_supplement.py` (Figures 1–7 and the Dense/random/stress/PAM50 sensitivity figures). |

## 4. Aggregated results

| File | Analysis |
| --- | --- |
| `data/results/comprehensive_batch1_single_omics_results.tsv` | Full-sample single-omics five-fold CV results. |
| `data/results/comprehensive_intersection_single_omics_results.tsv` | Intersection-sample single-omics baselines. |
| `data/results/comprehensive_triple_integration_results.tsv` | Triple-omics fusion results. |
| `data/results/comprehensive_pairwise_integration_results.tsv` | Pairwise-omics fusion results. |
| `data/results/final_9model_stacking_results.tsv` | Nine-model Stacking results. |
| `data/results/fullsize_multimodal_integration_results.tsv` | FullSizeCNN multi-omics integration results. |
| `data/results/fullsize_multimodal_integration_paired_tests.tsv` | Paired tests for FullSizeCNN integration. |
| `data/results/advanced_method*.tsv` | Multi-task, DeepSurv, Transformer, low-rank bilinear, and GNN results. |
| `data/results/scheme2_category_masking_results.tsv` | Scheme-2 functional-category masking results. |
| `data/results/other_omics_category_masking_results.tsv` | CNV/miRNA functional-category masking results. |
| `data/results/mrna_equal_mask_control_paired_tests.tsv` | Equal-size random masking control tests. |
| `data/dense_equivalent_baseline_results.tsv` | Dense-equivalent baseline versus FullSizeCNN comparison. |
| `data/random_permutation_control_results.tsv` | Per-repeat random-permutation ordering control results. |
| `data/random_permutation_control_stats.tsv` | Summary statistics and empirical p-values for the ordering control. |
| `data/stress_pathway_analysis_results.tsv` | Stress / adaptive-reprogramming pathway enrichment and survival analysis. |
| `data/pam50_gene_exclusion_results.tsv` | PAM50 50-gene exclusion sensitivity analysis. |

## 5. Interpretability outputs

| Path | Description |
| --- | --- |
| `data/interpretability/*_shap_importance.tsv` | SHAP feature importance for mRNA PAM50. |
| `data/interpretability/*_cnn_saliency_importance.tsv` | CNN saliency feature importance. |
| `data/interpretability/*_consensus_key_factors.tsv` | Consensus key factors between SHAP and saliency. |
| `data/interpretability/survival_key_factors.tsv` | Univariate survival key factors. |
| `data/interpretability/mrna_pam50_top100_enrichment.tsv` | Top-100 gene enrichment. |
| `data/interpretability/mrna_pam50_other_reenrichment.tsv` | Secondary enrichment for Other genes. |
| `data/interpretability/figures/` | SHAP, saliency, enrichment, and masking figures, including the final `fig_scheme2_masking_flow_en_v2.png` and `fig_other_omics_category_masking_v3.png` versions. |

## 6. Code

All runnable Python scripts are stored under `scripts/`. Key scripts include:

- `build_manuscript_and_supplement.py` — rebuilds the manuscript and supplementary Word documents from result tables and figures.
- `run_comprehensive_batch1_single_omics.py` — single-omics ML/MLP/FullSizeCNN baselines.
- `run_comprehensive_intersection_single_omics.py` — intersection single-omics baselines.
- `run_comprehensive_triple_integration.py` — triple-omics fusion.
- `run_comprehensive_pairwise_integration.py` — pairwise-omics fusion.
- `run_fullsize_multimodal_integration.py` — FullSizeCNN multimodal integration and paired tests.
- `run_advanced_method*.py` — advanced deep integration methods.
- `run_interpretability_key_factors.py` — SHAP, saliency, and survival key factors.
- `run_scheme2_category_masking.py`, `run_other_omics_category_masking.py`, and `run_mrna_equal_mask_control.py` — functional masking experiments.
- `generate_nsre_images.py` and `adaptive_nsre.py` — JSD scoring and omics image generation.
- `run_dense_baseline.py` — Dense-equivalent baseline analysis.
- `run_random_permutation_control.py` — random-permutation ordering control.
- `run_stress_pathway_analysis.py` — stress / adaptive-reprogramming pathway analysis.
- `pam50_gene_exclusion_sensitivity.py` — PAM50 50-gene exclusion sensitivity analysis.
- `generate_annotation_mapping.py` and `generate_sample_provenance.py` — annotation and provenance files used by the manuscript.

## 7. Machine-readable supplementary data and final drafts

| Path | Description |
| --- | --- |
| `data/supplementary_data/supplementary_data_tables.zip` | CSV copies of all `data/**/*.tsv` result tables plus `manifest.tsv`. |
| `docs/Supervised_distribution_divergence_guided_omics_imaging_manuscript_CN.docx` | Final Chinese manuscript draft. |
| `docs/Supervised_distribution_divergence_guided_omics_imaging_supplementary_material_CN.docx` | Final Chinese supplementary-material draft. |

## Reproducibility notes

- Use Python 3.11 and install the packages listed in the repository README.
- Keep `RANDOM_STATE=42`.
- Run analyses from the repository root, with `data/` and `scripts/` in their original relative locations.
