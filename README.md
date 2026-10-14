# Supervised distribution-divergence-guided omics imaging and multi-omics fusion for breast cancer molecular subtype reconstruction and survival risk modeling

This repository provides the analysis code, modeling-ready feature datasets, JSD image arrays, selected result tables, and interpretability outputs for the manuscript:

**Supervised distribution-divergence-guided omics imaging and multi-omics fusion for breast cancer molecular subtype reconstruction and survival risk modeling**

The repository is intended for research reproducibility. TCGA-BRCA raw omics data are public and can be obtained from GDC and UCSC Xena; only the processed modeling-ready tables and image arrays generated for this study are included here.

## Repository layout

- `scripts/`  
  Python scripts for data alignment, feature selection, JSD scoring, omics imaging, single-omics modeling, multi-omics fusion, statistical tests, and interpretability analysis.
- `docs/`  
  Final Chinese manuscript and supplementary-material drafts:
  - `docs/Supervised_distribution_divergence_guided_omics_imaging_manuscript_CN.docx`
  - `docs/Supervised_distribution_divergence_guided_omics_imaging_supplementary_material_CN.docx`
- `data/brca_labels_modeling_ready.tsv`  
  Modeling-ready clinical/molecular labels: case ID, PAM50 four-class label, OS event, and OS time.
- `data/final_datasets/`  
  Final feature matrices for PAM50 and survival tasks (features selected within each cross-validation training fold).
- `data/selected_features/`  
  Feature names and per-omics selected feature matrices.
- `data/images/`  
  JSD-ordered grayscale images and category/JSD grids.
- `data/results/`  
  Aggregated five-fold cross-validation results for single-omics, pairwise fusion, triple fusion, advanced methods, and statistical tests.
- `data/supplementary_data/supplementary_data_tables.zip`  
  Machine-readable CSV copies of all `data/**/*.tsv` result tables, together with `manifest.tsv` mapping each source table to its exported CSV.
- `data/interpretability/`  
  SHAP, saliency, consensus gene, pathway enrichment, and functional-masking outputs.
- `DATA_AND_CODE_MANIFEST.md`  
  Detailed file manifest and reproducibility notes.

## Data availability

The raw TCGA-BRCA data are available from:

- Genomic Data Commons: https://portal.gdc.cancer.gov/projects/TCGA-BRCA
- UCSC Xena: https://xenabrowser.net/

## Software and environment

- Python 3.11
- PyTorch
- scikit-learn
- pandas, NumPy, matplotlib
- lifelines

## Reproducibility

Random seed 42 was used for the analyses. All feature selection, scaling, augmentation, and model fitting were performed within cross-validation training folds; test folds were not used for preprocessing or parameter selection.

## Important update: leak-free re-analysis (2026-10)

An audit of the earlier pipeline identified that feature selection and JSD
ordering had been performed on the full cohort before cross-validation, which
could leak test-fold labels into feature choice and inflate the reported
five-fold estimates. We therefore re-ran all single-omics, pairwise, triple-omics
fusion, and advanced-method analyses with feature selection, scaling, JSD
ordering, and image generation performed strictly inside each training fold.

The corrected leak-free results are provided under `data/results/leakfree_*.tsv`:

- `leakfree_single_omics_results.tsv` — mRNA and miRNA single-omics (leak-free).
- `leakfree_cnv_single_rerun.tsv` — CNV single-omics using the corrected GDC gene-level CNV matrix.
- `leakfree_intersection_single_full.tsv` — single-omics baselines on the triple-omics intersection.
- `leakfree_pairwise_fusion_full.tsv` — pairwise-omics fusion.
- `leakfree_cnv_fusion_rerun.tsv` — triple-omics fusion.
- `leakfree_advanced_results.tsv` — Transformer and DeepSurv.
- `leakfree_survival_methods.tsv` — CoxNet, random survival forest, and time-dependent AUC.

The corrected gene-level CNV matrix (case_id x genes, thresholded to -2/-1/0/1/2)
is provided as `data/cnv_gene_case_matrix.tsv.gz`, rebuilt from GDC Gene Level
Copy Number files (`scripts/download_gdc_cnv.py`,
`scripts/build_gdc_cnv_matrix.py`, `scripts/finalize_cnv_matrix.py`).

The original cohort-level (exploratory) results remain available under
`data/results/` for historical comparison but should be treated as optimistic
estimates. The English manuscript and supplementary material reflect the
leak-free results (`docs/*_EN.docx`).

## License

Code and processed data are provided for academic research use. See `LICENSE` for the current terms.
