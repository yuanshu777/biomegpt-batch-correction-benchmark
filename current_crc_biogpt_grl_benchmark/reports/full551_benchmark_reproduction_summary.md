# Full551 MMUPHin Benchmark Reproduction

This restores the original 551-sample MMUPHin CRC dataset as the canonical benchmark. Metrics are copied from the frozen R controlled benchmark evaluator.

## Raw vs MMUPHin Metrics

| metric                             |    mmuphin |        raw |
|:-----------------------------------|-----------:|-----------:|
| condition_R2_study_controlled      | 0.00884871 | 0.00787524 |
| condition_R2_unadjusted_model      | 0.0092378  | 0.009224   |
| disease_LOSO_balanced_accuracy     | 0.623513   | 0.615319   |
| disease_LOSO_mean_within_study_AUC | 0.687841   | 0.709835   |
| disease_LOSO_overall_AUC           | 0.679738   | 0.690854   |
| study_R2_condition_controlled      | 0.0300456  | 0.0785634  |
| study_R2_unadjusted_model          | 0.0304347  | 0.0799121  |
| study_prediction_accuracy          | 0.671506   | 0.77193    |
| study_prediction_balanced_accuracy | 0.673665   | 0.75637    |

## Outputs

- `outputs/metrics/full551_raw_mmuphin_metrics.csv`
- `outputs/figures/full551_raw_mmuphin_pca_by_study.png`
- `outputs/figures/full551_raw_mmuphin_pca_by_condition.png`
- `outputs/figures/full551_raw_mmuphin_pca_scores.csv`
- `outputs/figures/full551_raw_pca_by_study.png`
- `outputs/figures/full551_mmuphin_pca_by_study.png`
- `outputs/figures/full551_raw_pca_by_condition.png`
- `outputs/figures/full551_mmuphin_pca_by_condition.png`
