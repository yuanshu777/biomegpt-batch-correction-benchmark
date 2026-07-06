# GRL Abundance LOSO Correction

## Scope

For each held-out study, GRL correction was trained only on the other studies, then used to transform both train studies and the held-out study. Disease classification was then trained on corrected train studies and tested on the corrected held-out study.

This directly checks whether the high full-data disease AUC was caused by condition-label amplification during correction training.

## Summary

| method                                            |   disease_LOSO_mean_within_study_AUC |   disease_LOSO_overall_AUC |   disease_LOSO_balanced_accuracy |
|:--------------------------------------------------|-------------------------------------:|---------------------------:|---------------------------------:|
| raw_python_loso_classifier                        |                             0.74365  |                   0.7599   |                         0.692255 |
| mmuphin_python_loso_classifier                    |                             0.749864 |                   0.770037 |                         0.702769 |
| loso_grl_abundance_l8_lam10_cw01_rw1              |                             0.611565 |                   0.619019 |                         0.575119 |
| loso_grl_abundance_l8_lam10_cw001_rw1             |                             0.602399 |                   0.593841 |                         0.557322 |
| loso_grl_abundance_l8_lam10_cw0_rw1               |                             0.555455 |                   0.571068 |                         0.549141 |
| loso_grl_mech_context_only_l8_lam10_rw5_rel1_var1 |                             0.612939 |                   0.609318 |                         0.593246 |

## Reading

- Compare GRL rows to the Python raw/MMUPHin rows, not directly to the R/glmnet values.
- In this run, all three LOSO-corrected GRL settings underperform the raw/MMUPHin Python LOSO baselines on disease AUC.
- Because `cw0.1` loses its high full-data AUC here, the full-data 0.882 result is best treated as transductive condition amplification, not reliable disease preservation.

## Output Files

- `summary`: `outputs\metrics\grl_abundance_loso_correction_summary.csv`
- `per_fold_metrics`: `outputs\metrics\grl_abundance_loso_correction_per_fold_metrics.csv`
- `training_history`: `outputs\metrics\grl_abundance_loso_correction_training_history.csv`
- `oof_matrices`: `outputs\crc_full_abundance_grl_loso`
