# GRL CRC389 Tuning Sweep

## Scope

This is a small local tuning sweep for the abundance-level GRL prototype on the MMUPHin CRC 389-overlap benchmark. It does not run BiomeGPT, does not use A100, and is not a final scGPT/BiomeGPT result.

## Why This Sweep

The first quick GRL run preserved CRC/control too strongly and increased external study predictability. This sweep tests condition-aware study adversaries, weaker condition preservation, weaker preservation loss, stronger GRL pressure, and a study-conditioned decoder.

## Baseline Reference

| method                     |   n_samples |   n_features_or_dims |   study_balanced_accuracy |   study_macro_f1 |   condition_auc |   condition_balanced_accuracy |   condition_macro_f1 | study_permanova_r2   | condition_permanova_r2   |   study_centroid_r2_fallback |   condition_centroid_r2_fallback |
|:---------------------------|------------:|---------------------:|--------------------------:|-----------------:|----------------:|------------------------------:|---------------------:|:---------------------|:-------------------------|-----------------------------:|---------------------------------:|
| Raw abundance              |         389 |                  484 |                  0.62069  |         0.618149 |        0.71344  |                      0.655241 |             0.654279 |                      |                          |                    0.0306354 |                       0.00683636 |
| MMUPHin adjusted abundance |         389 |                  484 |                  0.465169 |         0.46271  |        0.748663 |                      0.676715 |             0.675253 |                      |                          |                    0.0121529 |                       0.00807908 |

## Sweep Results

| case_id                                      |   lambda_grl |   latent_dim | lambda_schedule   |   warmup_fraction |   condition_weight |   preserve_weight | condition_aware_adversary   | use_study_conditioned_decoder   |   n_samples |   n_features_or_dims |   study_balanced_accuracy |   study_macro_f1 |   condition_auc |   condition_balanced_accuracy |   condition_macro_f1 |   study_centroid_r2_fallback |   condition_centroid_r2_fallback |   final_internal_study_balanced_accuracy |   final_internal_condition_auroc |
|:---------------------------------------------|-------------:|-------------:|:------------------|------------------:|-------------------:|------------------:|:----------------------------|:--------------------------------|------------:|---------------------:|--------------------------:|-----------------:|----------------:|------------------------------:|---------------------:|-----------------------------:|---------------------------------:|-----------------------------------------:|---------------------------------:|
| A_condaware_cw05_pw001_lam1                  |            1 |           64 | linear            |               0.1 |                0.5 |             0.01  | True                        | False                           |         389 |                   64 |                  0.821064 |         0.813891 |        1        |                      0.998092 |             0.997083 |                    0.08095   |                        0.0279193 |                                 0.825527 |                         1        |
| B_condaware_cw02_pw001_lam2                  |            2 |           64 | linear            |               0.1 |                0.2 |             0.01  | True                        | False                           |         389 |                   64 |                  0.782247 |         0.771761 |        0.99985  |                      0.994275 |             0.991284 |                    0.0739615 |                        0.0197331 |                                 0.789088 |                         0.99979  |
| C_condaware_cw05_pw000_lam2                  |            2 |           64 | linear            |               0.1 |                0.5 |             0     | True                        | False                           |         389 |                   64 |                  0.793355 |         0.784154 |        1        |                      0.998092 |             0.997083 |                    0.0806831 |                        0.0259254 |                                 0.781986 |                         1        |
| D_condaware_cw05_pw001_lam5_studydecoder     |            5 |           64 | linear            |               0.1 |                0.5 |             0.01  | True                        | True                            |         389 |                   64 |                  0.780282 |         0.774187 |        0.99991  |                      0.998092 |             0.997083 |                    0.0767424 |                        0.0230349 |                                 0.786235 |                         0.999699 |
| E_bottleneck16_no_condition_no_preserve_lam5 |            5 |           16 | linear            |               0.1 |                0   |             0     | True                        | False                           |         389 |                   16 |                  0.670711 |         0.648445 |        0.649456 |                      0.610281 |             0.594093 |                    0.0831058 |                        0.0052396 |                                 0.698629 |                         0.64026  |
| F_bottleneck16_cw01_no_preserve_lam10        |           10 |           16 | linear            |               0.1 |                0.1 |             0     | True                        | False                           |         389 |                   16 |                  0.730751 |         0.707647 |        0.877081 |                      0.788619 |             0.773676 |                    0.0826995 |                        0.0176698 |                                 0.748351 |                         0.868486 |
| G_bottleneck8_cw01_pw0001_lam10              |           10 |            8 | linear            |               0.1 |                0.1 |             0.001 | True                        | False                           |         389 |                    8 |                  0.453284 |         0.417404 |        0.802158 |                      0.740308 |             0.721988 |                    0.0453608 |                        0.0182251 |                                 0.446923 |                         0.788904 |
| H_bottleneck8_no_condition_no_preserve_lam10 |           10 |            8 | linear            |               0.1 |                0   |             0     | True                        | False                           |         389 |                    8 |                  0.599492 |         0.558519 |        0.678848 |                      0.635812 |             0.615296 |                    0.0775705 |                        0.017707  |                                 0.61468  |                         0.654325 |

## Best Local Setting

- Selected case: `G_bottleneck8_cw01_pw0001_lam10`
- Study balanced accuracy: 0.453 versus raw 0.621 and MMUPHin 0.465.
- CRC/control AUROC: 0.802 versus raw 0.713 and MMUPHin 0.749.
- Did it reduce study predictability versus raw? Yes.
- Did it keep CRC/control AUROC within 0.05 of raw? Yes.

## Interpretation

- Treat this as local objective debugging only. The representation was trained on all 389 samples with labels, and external probes are cross-validated on the learned representation.
- If none of the cases reduce study predictability below raw abundance, the current GRL objective is still not a controlled correction method for this abundance benchmark.
- If a case reduces study predictability while preserving condition signal, it is a candidate scaffold for a clearly labeled BiomeGPT CLS smoke check, not a final result.

## Output Files

- `baseline_metrics`: `outputs\metrics\grl_crc389_tuning_baseline.csv`
- `sweep_metrics`: `outputs\metrics\grl_crc389_tuning_sweep.csv`
- `best_config`: `outputs\metrics\grl_crc389_tuning_best.json`
- `best_corrected_representation`: `outputs\crc_overlap_benchmark\grl_tuned_best_raw_abundance_z_389.csv`
- `sweep_result_dir`: `outputs\crc_overlap_benchmark\grl_tuning_sweep_crc389`
- `best_figure_dir`: `outputs\figures\grl_tuning_crc389`
