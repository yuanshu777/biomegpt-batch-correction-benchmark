# Condition Amplification Diagnostics on Full CRC

## Scope

Diagnostics for raw abundance, MMUPHin, and GRL abundance-decoder outputs. The goal is to see whether high disease AUC is accompanied by artificial condition-axis amplification or low-dimensional collapse.

## Summary

| method        |   effective_rank |   dims_for_90pct_variance |   pc1_variance_fraction |   pc1_pc2_variance_fraction |   pc1_condition_auc_abs |   pc1_condition_correlation |   condition_R2_study_controlled_linear |   study_R2_condition_controlled_linear |   sample_sum_min |   sample_sum_max |   zero_fraction |
|:--------------|-----------------:|--------------------------:|------------------------:|----------------------------:|------------------------:|----------------------------:|---------------------------------------:|---------------------------------------:|-----------------:|-----------------:|----------------:|
| raw           |         221.731  |                       213 |               0.0327213 |                   0.0620517 |                0.626236 |                   0.201955  |                             0.00591663 |                              0.0356587 |         0.566068 |                1 |        0.767661 |
| mmuphin       |         224.303  |                       214 |               0.0314922 |                   0.0599072 |                0.521729 |                   0.036865  |                             0.00625939 |                              0.0223689 |         0.566068 |                1 |        0.767661 |
| grl_cw01      |          10.393  |                        10 |               0.27629   |                   0.48499   |                0.53443  |                   0.0566515 |                             0.0230087  |                              0.0582821 |         0.566068 |                1 |        0.157932 |
| grl_cw001     |          10.1033 |                         9 |               0.295408  |                   0.491535  |                0.630069 |                   0.221741  |                             0.0184329  |                              0.061913  |         0.566068 |                1 |        0.128774 |
| grl_cw0       |          10.0791 |                         9 |               0.288553  |                   0.475267  |                0.53603  |                   0.052086  |                             0.0109366  |                              0.0475022 |         0.566068 |                1 |        0.129712 |
| grl_mech_best |          24.2552 |                        23 |               0.168018  |                   0.29694   |                0.587166 |                   0.120577  |                             0.0131613  |                              0.0125931 |         0.566068 |                1 |        0.304724 |

## Reading

- A high `pc1_condition_auc_abs`, very high condition R2, or very small effective rank would suggest condition-coded amplification.
- Compare `grl_cw01` to `grl_cw001` and `grl_cw0`: if stronger condition supervision increases condition R2 sharply, it should be treated cautiously.

## Output Files

- `diagnostics`: `outputs\metrics\condition_amplification_full_crc_diagnostics.csv`
- `top_condition_features`: `outputs\metrics\condition_amplification_full_crc_top_features.csv`
- `figures`: `outputs\figures\condition_amplification_full_crc`
