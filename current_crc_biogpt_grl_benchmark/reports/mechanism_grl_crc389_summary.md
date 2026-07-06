# Mechanism-Only GRL on CRC389 Overlap

## Scope

This trains the mechanism-only abundance decoder on the 389 CRC overlap samples so the abundance result can be compared with BiomeGPT CLS baselines on the same sample set.

This is a CRC389 diagnostic table, not the original full 551-sample MMUPHin R benchmark.

## Primary Table

| method                           |   condition_R2_study_controlled |   disease_LOSO_mean_within_study_AUC |   study_R2_condition_controlled |   study_prediction_balanced_accuracy |
|:---------------------------------|--------------------------------:|-------------------------------------:|--------------------------------:|-------------------------------------:|
| Raw abundance                    |                      0.00710953 |                             0.733621 |                     0.0272079   |                            0.795247  |
| MMUPHin adjusted abundance       |                      0.00728463 |                             0.776142 |                     0.0160286   |                            0.569473  |
| Mechanism-only GRL abundance     |                      0.0176364  |                             0.685878 |                     0.0170376   |                            0.38201   |
| BiomeGPT raw CLS                 |                      0.00925751 |                             0.686386 |                     0.0572406   |                            0.71193   |
| BiomeGPT study-mean-centered CLS |                      0.00985056 |                             0.694251 |                     0.000695741 |                            0.0487899 |

## Reading

- Mechanism-only GRL abundance Study BA is 0.382 versus MMUPHin 0.569.
- Mechanism-only GRL abundance disease LOSO AUC is 0.686 versus MMUPHin 0.776.
- This table is mainly for same-sample linkage to BiomeGPT CLS; final MMUPHin-method claims should still use the full 551-sample R benchmark.

## Output Files

- `mechanism_grl_abundance`: `outputs\crc_overlap_benchmark\mechanism_grl_abundance_389.csv`
- `primary_table`: `outputs\metrics\mechanism_grl_crc389_primary_table.csv`
- `metrics_long`: `outputs\metrics\mechanism_grl_crc389_metrics_long.csv`
- `training_history`: `outputs\metrics\mechanism_grl_crc389_training_history.csv`
- `figures`: `outputs\figures\mechanism_grl_crc389`
