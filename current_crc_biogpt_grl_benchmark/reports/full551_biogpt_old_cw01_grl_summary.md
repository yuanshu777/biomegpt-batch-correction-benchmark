# Full551 BiomeGPT Old cw0.1-Style GRL

This runs the old supervised cw0.1 GRL idea on top of BiomeGPT raw CLS embeddings for the canonical 551 MMUPHin CRC samples. It is an embedding-level correction prototype, not foundation-model pretraining.

Config summary: `latent_dim=8`, `lambda_grl=10`, `condition_weight=0.1`, `preserve_weight=0.001`, linear warmup, condition-aware study adversary.

## Primary Table

| method                                   |   condition_R2_study_controlled |   disease_LOSO_mean_within_study_AUC |   study_R2_condition_controlled |   study_prediction_balanced_accuracy |
|:-----------------------------------------|--------------------------------:|-------------------------------------:|--------------------------------:|-------------------------------------:|
| Raw abundance                            |                      0.00591663 |                             0.73995  |                     0.0356587   |                            0.837985  |
| MMUPHin adjusted abundance               |                      0.00625939 |                             0.754655 |                     0.0223689   |                            0.720998  |
| BiomeGPT raw CLS 551                     |                      0.0110283  |                             0.694284 |                     0.15781     |                            0.683915  |
| BiomeGPT study-mean-centered CLS 551     |                      0.0131933  |                             0.69583  |                     0.000195502 |                            0.0993311 |
| BiomeGPT old cw0.1-style GRL CLS 551     |                      0.0153429  |                             0.654554 |                     0.0991049   |                            0.45987   |
| BiomeGPT NoMean conditional GRL CLS 551  |                      0.0112457  |                             0.690606 |                     0.148916    |                            0.532958  |
| BiomeGPT split conditional CORAL CLS 551 |                      0.0183277  |                             0.674913 |                     0.0841793   |                            0.586778  |

## Reading

- Raw CLS study BA 0.684, disease LOSO AUC 0.694, study R2 0.1578.
- Old cw0.1-style CLS GRL study BA 0.460, disease LOSO AUC 0.655, study R2 0.0991.
- Because this uses a condition classifier and an 8-dimensional bottleneck, inspect effective rank and PC1 condition AUC before interpreting any disease AUC improvement.
- This is full-data/transductive correction. It cannot support a final claim without LOSO/cross-fitted correction.

## Outputs

- `outputs\crc_full551_benchmark\biogpt_old_cw01_grl_z8_cls_551.csv`
- `outputs\metrics\full551_biogpt_old_cw01_grl_comparison.csv`
- `outputs\metrics\full551_biogpt_old_cw01_grl_diagnostics.csv`
- `outputs\figures\full551_biogpt_old_cw01_grl`
