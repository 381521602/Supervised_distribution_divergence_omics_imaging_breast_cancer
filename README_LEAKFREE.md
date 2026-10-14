# Leak-free re-analysis — reproducibility guide

This guide documents how the leak-free results in `data/results/leakfree_*.tsv`
were produced, so reviewers can reproduce them end to end.

## Why this re-analysis exists

The earlier pipeline selected features and computed JSD ordering on the full
cohort before cross-validation, which can leak test-fold labels into feature
choice. The corrected pipeline performs feature selection, scaling, JSD
ordering, and image generation strictly inside each training fold.

## Inputs

- Labels: `data/brca_labels_modeling_ready.tsv`
- Raw mRNA: `data/external/xena/HiSeqV2` (gene x sample)
- Raw miRNA: `data/external/xena/miRNA_HiSeq_gene` (MIMAT x sample)
- Corrected CNV: `data/cnv_gene_case_matrix.tsv` (case_id x gene, integer copy
  number discretized to -2/-1/0/1/2 by the standard convention
  0->-2, 1->-1, 2->0, 3->+1, >=4->+2, missing->0)

The corrected CNV matrix is rebuilt from GDC Gene Level Copy Number files:

```bash
python scripts/download_gdc_cnv.py
python scripts/build_gdc_cnv_matrix.py
python scripts/finalize_cnv_matrix.py
```

## Fixed split

All analyses use 5-fold stratified cross-validation with
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Feature-number
selection and evaluation share the same split, so the single-omics estimates are
leak-free but not fully nested; the nested repeated CV is a separate robustness
check (see `scripts/run_additional_validation_experiments.py`).

## Feature-selection method per omics and task

Defined in `scripts/adaptive_jsd.py` (`build_adaptive_pipeline`):

| Omics | PAM50 (k) | Survival (k) |
|-------|-----------|--------------|
| mRNA | FClassif -> JSD (400) | L1 -> JSD (200) |
| CNV | L1 -> one-vs-rest JSD (50) | L1 -> JSD (150) |
| miRNA | JSD -> L1 (600) | L1 -> JSD (600) |

Image sizes: PAM50 mRNA 20, CNV 8, miRNA 25; survival mRNA 15, CNV 13, miRNA 25.

## Run the leak-free analyses

```bash
# single-omics (mRNA, miRNA; CNV is run separately with the corrected matrix)
python scripts/run_leakfree_single_omics.py

# corrected-CNV single-omics
python scripts/rerun_cnv_single.py

# intersection single-omics (full model set) and pairwise fusion
python scripts/run_full_s5_s7.py

# triple-omics fusion (corrected CNV)
python scripts/rerun_cnv_fusion.py

# advanced methods (Transformer, DeepSurv)
python scripts/run_leakfree_advanced.py

# CoxNet / random survival forest / time-dependent AUC
python scripts/run_survival_methods.py

# paired t-test / Wilcoxon on same-fold predictions
python scripts/run_paired_tests.py

# out-of-fold predictions (per patient, per fold)
python scripts/run_oof_predictions.py
```

## Outputs

- `data/results/leakfree_single_omics_results.tsv`
- `data/results/leakfree_cnv_single_rerun.tsv`
- `data/results/leakfree_intersection_single_full.tsv`
- `data/results/leakfree_pairwise_fusion_full.tsv`
- `data/results/leakfree_cnv_fusion_rerun.tsv`
- `data/results/leakfree_advanced_results.tsv`
- `data/results/leakfree_survival_methods.tsv`
- `data/results/leakfree_paired_tests.tsv`
- `data/leakfree_oof_predictions.tsv`

The old cohort-level (exploratory) results remain under `data/results/` for
historical comparison only and should not be cited as unbiased estimates.

## Known provenance caveats

- The corrected CNV matrix and the raw miRNA matrix have slightly smaller sample
  coverage than the original cohort manifest, so the triple-omics intersections
  are 486 (PAM50) and 725 (survival), not the earlier 493/731.
- The integer copy-number discretization is a standard convention, not the
  GISTIC2 algorithm; it agrees with Xena GISTIC2 thresholded calls on 50.6% of
  overlapping gene-sample cells (Cohen's kappa 0.257, Pearson r 0.565).
