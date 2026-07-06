# CLS Architecture Ablation CRC389

## Scope

This run tests architecture changes after the NoMean residual adapter failed to move study R2 enough. It does not train BiomeGPT, does not apply study mean-centering inside the tested methods, and does not optimize final evaluator metrics directly.

Tested candidates:

- H: fixed linear study-subspace projection from raw CLS
- G: invariant/nuisance split adapter with weak conditional GRL
- I: invariant/nuisance split adapter with weak conditional GRL plus conditional CORAL

## Primary Table

| method                                         |   condition_R2_study_controlled |   disease_LOSO_mean_within_study_AUC |   study_R2_condition_controlled |   study_prediction_balanced_accuracy |
|:-----------------------------------------------|--------------------------------:|-------------------------------------:|--------------------------------:|-------------------------------------:|
| Raw abundance                                  |                      0.00710953 |                             0.733621 |                     0.0272079   |                            0.795247  |
| MMUPHin adjusted abundance                     |                      0.00728463 |                             0.776142 |                     0.0160286   |                            0.569473  |
| BiomeGPT raw CLS                               |                      0.00925751 |                             0.686386 |                     0.0572406   |                            0.71193   |
| BiomeGPT study-mean-centered CLS               |                      0.00985056 |                             0.694251 |                     0.000695741 |                            0.0487899 |
| A NoMean adapter + vanilla GRL                 |                      0.00891115 |                             0.670376 |                     0.0507143   |                            0.461016  |
| B NoMean adapter + conditional GRL             |                      0.0093673  |                             0.689491 |                     0.0509044   |                            0.513215  |
| H study-subspace projection from raw CLS       |                      0.00930957 |                             0.688572 |                     0.0551121   |                            0.53205   |
| G invariant/nuisance split + weak GRL          |                      0.0133694  |                             0.643033 |                     0.0601302   |                            0.610286  |
| I invariant/nuisance split + conditional CORAL |                      0.0107589  |                             0.682583 |                     0.0283911   |                            0.662469  |

## H/G/I Tradeoff Table

| method                                         |   condition_R2_study_controlled |   disease_LOSO_mean_within_study_AUC |   study_R2_condition_controlled |   study_prediction_balanced_accuracy |   batch_reduction_vs_raw_cls |   study_r2_reduction_vs_raw_cls |   biology_change_vs_raw_cls |   biology_retention_vs_raw_cls |
|:-----------------------------------------------|--------------------------------:|-------------------------------------:|--------------------------------:|-------------------------------------:|-----------------------------:|--------------------------------:|----------------------------:|-------------------------------:|
| H study-subspace projection from raw CLS       |                      0.00930957 |                             0.688572 |                       0.0551121 |                             0.53205  |                    0.17988   |                      0.00212845 |                  0.00218582 |                       1.00318  |
| G invariant/nuisance split + weak GRL          |                      0.0133694  |                             0.643033 |                       0.0601302 |                             0.610286 |                    0.101643  |                     -0.00288963 |                 -0.0433539  |                       0.936837 |
| I invariant/nuisance split + conditional CORAL |                      0.0107589  |                             0.682583 |                       0.0283911 |                             0.662469 |                    0.0494611 |                      0.0288495  |                 -0.00380337 |                       0.994459 |

## Diagnostics

| method                                         |   effective_rank |   raw_effective_rank |   pc1_condition_auc |   raw_pc1_condition_auc |   mean_squared_shift_standardized |   mean_l2_shift_standardized |   output_dim |   removed_subspace_rank |   study_classifier_training_accuracy |
|:-----------------------------------------------|-----------------:|---------------------:|--------------------:|------------------------:|----------------------------------:|-----------------------------:|-------------:|------------------------:|-------------------------------------:|
| H study-subspace projection from raw CLS       |          99.7675 |              101.035 |            0.555148 |                0.555178 |                        0.00407254 |                      1.29202 |          512 |                       3 |                                    1 |
| G invariant/nuisance split + weak GRL          |          30.9086 |              101.035 |            0.535072 |                0.555178 |                      nan          |                    nan       |          128 |                     nan |                                  nan |
| I invariant/nuisance split + conditional CORAL |          58.8395 |              101.035 |            0.560137 |                0.555178 |                      nan          |                    nan       |          128 |                     nan |                                  nan |

## Reading

- Raw CLS: study R2 0.0572, study BA 0.712, disease LOSO AUC 0.686.
- MMUPHin adjusted abundance reference: study R2 0.0160, study BA 0.569, disease LOSO AUC 0.776.
- Main verdict: I invariant/nuisance split + conditional CORAL is the best current H/G/I candidate under the useful rule (study R2 0.0284, study BA 0.662, disease AUC 0.683).
- H is the most direct diagnostic of whether raw CLS study structure is carried by linear study-discriminative directions.
- G/I are full-data prototypes. If one looks promising, the next required step is LOSO/cross-fitted correction.

## Output Files

- `metrics_long`: `outputs\metrics\cls_architecture_ablation_crc389_metrics_long.csv`
- `primary_table`: `outputs\metrics\cls_architecture_ablation_crc389_primary_table.csv`
- `training_history`: `outputs\metrics\cls_architecture_ablation_crc389_training_history.csv`
- `diagnostics`: `outputs\metrics\cls_architecture_ablation_crc389_diagnostics.csv`
- `configs`: `outputs\metrics\cls_architecture_ablation_crc389_configs.json`
- `figure_dir`: `outputs\figures\cls_architecture_ablation_crc389`
- `corrected_adapter_dir`: `outputs\crc_overlap_benchmark`
