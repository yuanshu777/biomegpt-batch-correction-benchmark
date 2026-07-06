# NoMean CLS GRL Formulation Ablation CRC389

## Scope

This run compares five GRL formulations on the same NoMean residual BiomeGPT CLS adapter. It does not train BiomeGPT, does not use mean-centering inside the adapter, and does not optimize the final evaluator metrics directly.

All ablations use the same local prototype settings: 512-dimensional CLS output, residual scale 0.1, study-aware decoder reconstruction, preservation loss, anti-collapse variance/covariance penalties, and `lambda_grl=0.50` with linear warmup.

## Primary Table

| method                                                          |   condition_R2_study_controlled |   disease_LOSO_mean_within_study_AUC |   study_R2_condition_controlled |   study_prediction_balanced_accuracy |
|:----------------------------------------------------------------|--------------------------------:|-------------------------------------:|--------------------------------:|-------------------------------------:|
| Raw abundance                                                   |                      0.00710953 |                             0.733621 |                     0.0272079   |                            0.795247  |
| MMUPHin adjusted abundance                                      |                      0.00728463 |                             0.776142 |                     0.0160286   |                            0.569473  |
| BiomeGPT raw CLS                                                |                      0.00925751 |                             0.686386 |                     0.0572406   |                            0.71193   |
| BiomeGPT study-mean-centered CLS                                |                      0.00985056 |                             0.694251 |                     0.000695741 |                            0.0487899 |
| A NoMean adapter + vanilla GRL                                  |                      0.00891115 |                             0.670376 |                     0.0507143   |                            0.461016  |
| B NoMean adapter + conditional GRL                              |                      0.0093673  |                             0.689491 |                     0.0509044   |                            0.513215  |
| C NoMean adapter + residual conditional GRL                     |                      0.009373   |                             0.684457 |                     0.0561327   |                            0.5101    |
| D NoMean adapter + pairwise within-condition GRL                |                      0.00926923 |                             0.685664 |                     0.0569766   |                            0.711201  |
| E NoMean adapter + residual conditional GRL + conditional CORAL |                      0.00910908 |                             0.681695 |                     0.0501552   |                            0.509469  |

## Tradeoff Table

| method                                                          |   condition_R2_study_controlled |   disease_LOSO_mean_within_study_AUC |   study_R2_condition_controlled |   study_prediction_balanced_accuracy |   batch_reduction_vs_raw_cls |   biology_change_vs_raw_cls |   biology_retention_vs_raw_cls |
|:----------------------------------------------------------------|--------------------------------:|-------------------------------------:|--------------------------------:|-------------------------------------:|-----------------------------:|----------------------------:|-------------------------------:|
| A NoMean adapter + vanilla GRL                                  |                      0.00891115 |                             0.670376 |                       0.0507143 |                             0.461016 |                   0.250914   |                 -0.0160107  |                       0.976674 |
| B NoMean adapter + conditional GRL                              |                      0.0093673  |                             0.689491 |                       0.0509044 |                             0.513215 |                   0.198714   |                  0.00310502 |                       1.00452  |
| C NoMean adapter + residual conditional GRL                     |                      0.009373   |                             0.684457 |                       0.0561327 |                             0.5101   |                   0.20183    |                 -0.00192954 |                       0.997189 |
| D NoMean adapter + pairwise within-condition GRL                |                      0.00926923 |                             0.685664 |                       0.0569766 |                             0.711201 |                   0.00072871 |                 -0.00072228 |                       0.998948 |
| E NoMean adapter + residual conditional GRL + conditional CORAL |                      0.00910908 |                             0.681695 |                       0.0501552 |                             0.509469 |                   0.202461   |                 -0.00469112 |                       0.993165 |

## Adapter Diagnostics

| method                                                          |   effective_rank |   raw_effective_rank |   pc1_condition_auc |   raw_pc1_condition_auc |   mean_squared_shift_standardized |   mean_l2_shift_standardized |
|:----------------------------------------------------------------|-----------------:|---------------------:|--------------------:|------------------------:|----------------------------------:|-----------------------------:|
| A NoMean adapter + vanilla GRL                                  |          103.335 |              101.035 |            0.555779 |                0.555178 |                       0.0273412   |                     2.43471  |
| B NoMean adapter + conditional GRL                              |          101.774 |              101.035 |            0.557522 |                0.555178 |                       0.144055    |                     6.45215  |
| C NoMean adapter + residual conditional GRL                     |          102.353 |              101.035 |            0.557973 |                0.555178 |                       0.0142796   |                     1.48485  |
| D NoMean adapter + pairwise within-condition GRL                |          101.305 |              101.035 |            0.555208 |                0.555178 |                       9.95989e-05 |                     0.209696 |
| E NoMean adapter + residual conditional GRL + conditional CORAL |          103.145 |              101.035 |            0.549258 |                0.555178 |                       0.0420455   |                     2.42151  |

## Reading

- Raw CLS: study BA 0.712, study R2 0.0572, disease LOSO AUC 0.686.
- MMUPHin adjusted abundance reference: study BA 0.569, study R2 0.0160, disease LOSO AUC 0.776.
- Lowest study BA among ablations: A NoMean adapter + vanilla GRL with study BA 0.461.
- Main verdict: A NoMean adapter + vanilla GRL is the best current tradeoff under the raw-CLS preservation rule (study BA 0.461, disease LOSO AUC 0.670).
- This is still full-data/transductive adapter fitting. Any promising formulation needs LOSO/cross-fitted adapter correction before a scientific claim.

## Output Files

- `metrics_long`: `outputs\metrics\nomean_grl_ablation_crc389_metrics_long.csv`
- `primary_table`: `outputs\metrics\nomean_grl_ablation_crc389_primary_table.csv`
- `training_history`: `outputs\metrics\nomean_grl_ablation_crc389_training_history.csv`
- `diagnostics`: `outputs\metrics\nomean_grl_ablation_crc389_diagnostics.csv`
- `configs`: `outputs\metrics\nomean_grl_ablation_crc389_configs.json`
- `figure_dir`: `outputs\figures\nomean_grl_ablation_crc389`
- `corrected_adapter_dir`: `outputs\crc_overlap_benchmark`
