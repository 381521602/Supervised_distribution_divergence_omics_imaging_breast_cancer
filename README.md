# Supervised distribution-divergence-guided omics imaging and multi-omics fusion for breast cancer molecular subtyping and survival prediction

This repository provides the analysis code, modeling-ready feature datasets, JSD image arrays, selected result tables, and interpretability outputs for the manuscript:

**Supervised distribution-divergence-guided omics imaging and multi-omics fusion for breast cancer molecular subtyping and survival prediction**

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
  Final pre-selected feature matrices for PAM50 and survival tasks.
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

## License

Code and processed data are provided for academic research use. See `LICENSE` for the current terms.
