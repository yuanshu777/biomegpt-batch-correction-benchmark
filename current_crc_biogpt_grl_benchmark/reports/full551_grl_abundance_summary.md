# Full551 GRL Abundance Summary

This uses the original 551-sample MMUPHin CRC benchmark. Full-data GRL metrics are reported as diagnostics only; strict LOSO/OOF correction remains the robustness check.

## Primary Metrics

| method                                                |   condition_R2_study_controlled |   condition_R2_unadjusted_model |   disease_LOSO_balanced_accuracy |   disease_LOSO_mean_within_study_AUC |   disease_LOSO_overall_AUC |   study_R2_condition_controlled |   study_R2_unadjusted_model |   study_prediction_accuracy |   study_prediction_balanced_accuracy |
|:------------------------------------------------------|--------------------------------:|--------------------------------:|---------------------------------:|-------------------------------------:|---------------------------:|--------------------------------:|----------------------------:|----------------------------:|-------------------------------------:|
| grl_abundance_l8_lam10_cw001_rw1                      |                      0.0109302  |                       0.0114575 |                         0.59057  |                             0.667158 |                   0.631053 |                       0.0330634 |                   0.0335907 |                    0.54386  |                             0.529881 |
| grl_abundance_l8_lam10_cw01_rw1                       |                      0.0722728  |                       0.0775707 |                         0.777597 |                             0.882391 |                   0.854388 |                       0.0328371 |                   0.0381349 |                    0.54144  |                             0.553112 |
| grl_abundance_l8_lam10_cw0_rw1                        |                      0.0184733  |                       0.0224577 |                         0.627597 |                             0.674264 |                   0.643273 |                       0.0526877 |                   0.0566721 |                    0.528736 |                             0.506906 |
| grl_mech_context_only_l8_lam10_rw5_rel1_var1          |                      0.0233475  |                       0.0242185 |                         0.605022 |                             0.688281 |                   0.662662 |                       0.0114049 |                   0.0122758 |                    0.371446 |                             0.377597 |
| loso_grl_mech_context_only_l8_lam10_rw5_rel1_var1_oof |                      0.0115518  |                       0.0121495 |                         0.555386 |                             0.590779 |                   0.586823 |                       0.0599557 |                   0.0605534 |                    0.944344 |                             0.932084 |
| mmuphin                                               |                      0.00884871 |                       0.0092378 |                         0.623513 |                             0.687841 |                   0.679738 |                       0.0300456 |                   0.0304347 |                    0.671506 |                             0.673665 |
| raw                                                   |                      0.00787524 |                       0.009224  |                         0.615319 |                             0.709835 |                   0.690854 |                       0.0785634 |                   0.0799121 |                    0.77193  |                             0.75637  |

## Geometry Diagnostics

| method                                       |   n_samples |   n_features |   effective_rank |   python_standardized_pca_pc1_variance |   python_standardized_pca_pc2_variance |
|:---------------------------------------------|------------:|-------------:|-----------------:|---------------------------------------:|---------------------------------------:|
| raw                                          |         551 |          484 |         118.136  |                              0.0226038 |                              0.0211046 |
| mmuphin                                      |         551 |          484 |         113.858  |                              0.0230143 |                              0.0212837 |
| grl_abundance_l8_lam10_cw01_rw1              |         551 |          484 |          19.5855 |                              0.264801  |                              0.200887  |
| grl_abundance_l8_lam10_cw001_rw1             |         551 |          484 |          19.1543 |                              0.282425  |                              0.194942  |
| grl_abundance_l8_lam10_cw0_rw1               |         551 |          484 |          16.3754 |                              0.273777  |                              0.182648  |
| grl_mech_context_only_l8_lam10_rw5_rel1_var1 |         551 |          484 |          36.7237 |                              0.139844  |                              0.112517  |

## Interpretation

- Mechanism-only full-data GRL reduces study metrics strongly in the frozen R evaluator, but prior strict LOSO/OOF diagnostics did not preserve this advantage.
- High PC1/PC2 variance or low effective rank should be treated as possible artificial low-dimensional geometry.
