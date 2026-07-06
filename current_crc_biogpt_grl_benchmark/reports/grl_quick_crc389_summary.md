# GRL Quick CRC389 Sanity Check

## Scope

This is a quick local abundance-level GRL prototype on the MMUPHin CRC 389-overlap benchmark. It does not run BiomeGPT, does not use A100, and is not a final scGPT/BiomeGPT result.

## Inputs

- `outputs/crc_overlap_benchmark/raw_abundance_389.csv`
- `outputs/crc_overlap_benchmark/mmuphin_adjusted_abundance_389.csv`
- `outputs/crc_overlap_benchmark/metadata_389.csv`

## Methods Compared

- Raw abundance
- MMUPHin adjusted abundance
- GRL corrected representation trained from raw abundance features

## Metric Summary

| method                                     |   n_samples |   n_features_or_dims |   study_balanced_accuracy |   study_macro_f1 |   condition_auc |   condition_balanced_accuracy |   condition_macro_f1 | study_permanova_r2   | condition_permanova_r2   |   study_centroid_r2_fallback |   condition_centroid_r2_fallback |
|:-------------------------------------------|------------:|---------------------:|--------------------------:|-----------------:|----------------:|------------------------------:|---------------------:|:---------------------|:-------------------------|-----------------------------:|---------------------------------:|
| Raw abundance                              |         389 |                  484 |                  0.62069  |         0.618149 |        0.71344  |                      0.655241 |             0.654279 |                      |                          |                    0.0306354 |                       0.00683636 |
| MMUPHin adjusted abundance                 |         389 |                  484 |                  0.465169 |         0.46271  |        0.748663 |                      0.676715 |             0.675253 |                      |                          |                    0.0121529 |                       0.00807908 |
| GRL corrected raw-abundance representation |         389 |                   64 |                  0.79296  |         0.78819  |        0.99988  |                      0.994275 |             0.991284 |                      |                          |                    0.0578335 |                       0.0342148  |

## Interpretation

- Did GRL reduce study predictability compared with raw abundance? No. Study balanced accuracy changed from 0.621 to 0.793.
- Did GRL preserve CRC/control signal? Numerically yes. CRC/control AUROC changed from 0.713 to 1.000. Because the GRL representation was trained using all condition labels, the near-perfect condition probe should be read as a sanity-check signal, not strict held-out disease generalization.
- Compared with MMUPHin, GRL study balanced accuracy was higher by 0.328; CRC/control AUROC was higher by 0.251.
- Is this promising enough to connect to BiomeGPT CLS next? Not yet as a correction benchmark: this run increased study predictability. The next local step should be tuning the GRL objective on abundance features or running a clearly labeled CLS smoke check, not claiming improvement.

## Training Notes

- Epochs run: 100
- Final preservation loss: 0.706
- Final internal condition loss: 0.005
- Final internal study adversary loss: 1.281

## Limitations

- This is not BiomeGPT CLS yet.
- This is not full scGPT-style training.
- This uses abundance features as the input matrix.
- Final comparison should use raw BiomeGPT CLS and GRL-corrected BiomeGPT CLS on the same 389 samples.
- The external probes use cross-validation, but the learned GRL representation itself was fit on all 389 samples and their labels in this quick prototype.
- External probe metrics matter more than internal GRL training loss.
- `study_permanova_r2` and `condition_permanova_r2` are left blank because formal condition-controlled PERMANOVA is not implemented in this quick Python runner; centroid R2 fallback columns are provided only as local diagnostics.

## Output Files

- `baseline_metrics`: `outputs\metrics\grl_quick_baseline_raw_vs_mmuphin.csv`
- `method_comparison`: `outputs\metrics\grl_quick_method_comparison.csv`
- `grl_corrected_representation`: `outputs\crc_overlap_benchmark\grl_corrected_raw_abundance_z_389.csv`
- `grl_training_history`: `outputs\metrics\grl_training_history.csv`
- `grl_final_internal_metrics`: `outputs\metrics\grl_final_internal_metrics.csv`
- `figure_dir`: `outputs\figures\grl_quick_crc389`
- `save_grl_result_dir`: `outputs\crc_overlap_benchmark\grl_quick_raw_abundance`
