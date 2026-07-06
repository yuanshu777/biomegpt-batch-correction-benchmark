# Full551 BiomeGPT CLS Summary

This evaluates BiomeGPT CLS extracted directly for all 551 MMUPHin CRC samples, then runs local model-based CLS correction prototypes.

Important: abundance rows in this CLS table use the Python representation evaluator for same-table comparison. The frozen R/Bray-Curtis canonical abundance metrics remain in `full551_raw_mmuphin_metrics.csv` and `full551_grl_abundance_comparison.csv`.

## Primary Table

| method                                     |   condition_R2_study_controlled |   disease_LOSO_mean_within_study_AUC |   study_R2_condition_controlled |   study_prediction_balanced_accuracy |
|:-------------------------------------------|--------------------------------:|-------------------------------------:|--------------------------------:|-------------------------------------:|
| Raw abundance                              |                      0.00591663 |                             0.73995  |                     0.0356587   |                            0.837985  |
| MMUPHin adjusted abundance                 |                      0.00625939 |                             0.754655 |                     0.0223689   |                            0.720998  |
| BiomeGPT raw CLS 551                       |                      0.0110283  |                             0.694284 |                     0.15781     |                            0.683915  |
| BiomeGPT study-mean-centered CLS 551       |                      0.0131933  |                             0.69583  |                     0.000195502 |                            0.0993311 |
| BiomeGPT NoMean conditional GRL CLS 551    |                      0.0112457  |                             0.690606 |                     0.148916    |                            0.532958  |
| BiomeGPT study-subspace projection CLS 551 |                      0.0111576  |                             0.693676 |                     0.152589    |                            0.53259   |
| BiomeGPT split conditional CORAL CLS 551   |                      0.0183277  |                             0.674913 |                     0.0841793   |                            0.586778  |

## Reading

- Raw BiomeGPT CLS study R2 is 0.1578; study BA is 0.684; disease LOSO AUC is 0.694.
- These CLS corrections are full-data prototypes. Do not treat them as final without LOSO/cross-fitted correction.

## Outputs

- `outputs\metrics\full551_biogpt_cls_comparison.csv`
- `outputs\metrics\full551_biogpt_cls_diagnostics.csv`
- `outputs\figures\full551_biogpt_cls`
