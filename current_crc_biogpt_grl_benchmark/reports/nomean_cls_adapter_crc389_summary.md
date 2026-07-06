# NoMean BiomeGPT CLS Adapter CRC389 Prototype

## Scope

This is a local prototype on the 389 overlap samples. It does not train BiomeGPT, does not run scGPT pretraining, and does not apply manual study mean-centering inside the adapter method.

The adapter is a small residual module on raw BiomeGPT CLS: `z = h + 0.1 * Adapter(h)`. It keeps the CLS dimension at 512 and uses a study-aware decoder side-channel plus weak conditional study GRL. No condition classifier is used.

## Primary Table

| method                                       |   condition_R2_study_controlled |   disease_LOSO_mean_within_study_AUC |   study_R2_condition_controlled |   study_prediction_balanced_accuracy |
|:---------------------------------------------|--------------------------------:|-------------------------------------:|--------------------------------:|-------------------------------------:|
| Raw abundance                                |                      0.00710953 |                             0.733621 |                     0.0272079   |                            0.795247  |
| MMUPHin adjusted abundance                   |                      0.00728463 |                             0.776142 |                     0.0160286   |                            0.569473  |
| BiomeGPT raw CLS                             |                      0.00925751 |                             0.686386 |                     0.0572406   |                            0.71193   |
| BiomeGPT study-mean-centered CLS             |                      0.00985056 |                             0.694251 |                     0.000695741 |                            0.0487899 |
| NoMean CLS adapter preserve-only             |                      0.00926713 |                             0.685446 |                     0.0572124   |                            0.705305  |
| NoMean CLS adapter weak conditional GRL 0.05 |                      0.00926811 |                             0.685648 |                     0.0572036   |                            0.701872  |
| NoMean CLS adapter weak conditional GRL 0.10 |                      0.00926882 |                             0.685564 |                     0.0571848   |                            0.702891  |
| NoMean CLS adapter weak conditional GRL 0.30 |                      0.00928246 |                             0.682485 |                     0.0555679   |                            0.590236  |
| NoMean CLS adapter weak conditional GRL 0.50 |                      0.00953096 |                             0.682876 |                     0.0532628   |                            0.51239   |
| NoMean CLS adapter weak conditional GRL 1.00 |                      0.00914136 |                             0.673757 |                     0.050439    |                            0.554554  |

## Adapter Diagnostics

| method                                       |   effective_rank |   raw_effective_rank |   pc1_condition_auc |   raw_pc1_condition_auc |   mean_squared_shift_standardized |   mean_l2_shift_standardized |
|:---------------------------------------------|-----------------:|---------------------:|--------------------:|------------------------:|----------------------------------:|-----------------------------:|
| NoMean CLS adapter preserve-only             |          101.272 |              101.035 |            0.555238 |                0.555178 |                       8.32151e-05 |                     0.193556 |
| NoMean CLS adapter weak conditional GRL 0.05 |          101.27  |              101.035 |            0.555268 |                0.555178 |                       8.3702e-05  |                     0.194147 |
| NoMean CLS adapter weak conditional GRL 0.10 |          101.275 |              101.035 |            0.555208 |                0.555178 |                       8.76663e-05 |                     0.198471 |
| NoMean CLS adapter weak conditional GRL 0.30 |          101.61  |              101.035 |            0.555118 |                0.555178 |                       0.00150277  |                     0.453744 |
| NoMean CLS adapter weak conditional GRL 0.50 |          102.433 |              101.035 |            0.556561 |                0.555178 |                       0.00748615  |                     1.2293   |
| NoMean CLS adapter weak conditional GRL 1.00 |          103.508 |              101.035 |            0.553615 |                0.555178 |                       0.0127456   |                     1.72522  |

## Reading

- Raw CLS study BA is 0.712; mean-centering study BA is 0.049.
- Best adapter by study BA is `NoMean CLS adapter weak conditional GRL 0.50` with study BA 0.512 and disease LOSO AUC 0.683.
- Best adapter by disease LOSO AUC is `NoMean CLS adapter weak conditional GRL 0.05` with disease LOSO AUC 0.686 and study BA 0.702.
- Main verdict: NoMean CLS adapter weak conditional GRL 0.50 met the minimal useful rule: lower study BA than raw CLS and disease LOSO AUC within 0.05 of raw CLS.
- This is still full-data/transductive adapter fitting. If a configuration looks promising, the next strict check is LOSO/cross-fitted adapter correction.
- Do not claim it beats MMUPHin unless it survives the same-sample evaluator and strict held-out correction.

## Output Files

- `corrected_adapter_dir`: `outputs\crc_overlap_benchmark`
- `metrics_long`: `outputs\metrics\nomean_cls_adapter_crc389_metrics_long.csv`
- `primary_table`: `outputs\metrics\nomean_cls_adapter_crc389_primary_table.csv`
- `training_history`: `outputs\metrics\nomean_cls_adapter_crc389_training_history.csv`
- `diagnostics`: `outputs\metrics\nomean_cls_adapter_crc389_diagnostics.csv`
- `configs`: `outputs\metrics\nomean_cls_adapter_crc389_configs.json`
- `figure_dir`: `outputs\figures\nomean_cls_adapter_crc389`
