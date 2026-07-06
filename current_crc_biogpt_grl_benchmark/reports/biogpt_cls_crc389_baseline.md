# BiomeGPT CLS CRC389 Baseline

## Scope

This evaluates the stage2 BiomeGPT raw CLS embeddings for the 389 CRC overlap samples, then applies a simple study mean-centering baseline before any GRL-corrected CLS experiment.

The CLS matrix already existed at `outputs/crc_overlap_benchmark/biogpt_raw_cls_389.csv` and is recorded in the package reports as extracted from the stage2 checkpoint.

## Important Metric Note

The abundance methods and CLS methods are placed in the same Python MMUPHin-style diagnostic table. For CLS, R/Bray-Curtis abundance PERMANOVA is not directly applicable, so R2 is standardized Euclidean/linear partial R2 in representation space.

## Primary Table

| method                           |   condition_R2_study_controlled |   disease_LOSO_mean_within_study_AUC |   study_R2_condition_controlled |   study_prediction_balanced_accuracy |
|:---------------------------------|--------------------------------:|-------------------------------------:|--------------------------------:|-------------------------------------:|
| Raw abundance                    |                      0.00710953 |                             0.733621 |                     0.0272079   |                            0.795247  |
| MMUPHin adjusted abundance       |                      0.00728463 |                             0.776142 |                     0.0160286   |                            0.569473  |
| BiomeGPT raw CLS                 |                      0.00925751 |                             0.686386 |                     0.0572406   |                            0.71193   |
| BiomeGPT study-mean-centered CLS |                      0.00985056 |                             0.694251 |                     0.000695741 |                            0.0487899 |

## Mean-Centering Effect

- Study BA changed from 0.712 to 0.049.
- Disease LOSO AUC changed from 0.686 to 0.694.
- Study R2 changed from 0.0572 to 0.0007.
- Condition R2 changed from 0.0093 to 0.0099.

## Reading

- This is a baseline diagnostic, not a GRL/scGPT result.
- If mean-centering lowers study signal while preserving disease LOSO AUC, it is a useful simple baseline that GRL-corrected CLS must beat.
- If mean-centering hurts disease LOSO AUC, GRL needs an explicit preservation mechanism.

## Output Files

- `raw_cls`: `outputs\crc_overlap_benchmark\biogpt_raw_cls_389.csv`
- `mean_centered_cls`: `outputs\crc_overlap_benchmark\biogpt_mean_centered_cls_389.csv`
- `metrics_long`: `outputs\metrics\biogpt_cls_crc389_metrics_long.csv`
- `primary_table`: `outputs\metrics\biogpt_cls_crc389_primary_table.csv`
- `figure_dir`: `outputs\figures\biogpt_cls_crc389`
