# MMUPHin-Guided Residual GRL Full551

This rescue experiment uses MMUPHin-style feature-space residual correction instead of an 8D bottleneck reconstruction.

```text
x_corrected = x_raw + delta
```

Losses include conditional study GRL, abundance preservation, sqrt-relative-abundance preservation, covariance preservation, delta penalty, and optional MMUPHin anchor.

## R Evaluator Primary Metrics

| method                                             |   condition_R2_study_controlled |   disease_LOSO_balanced_accuracy |   disease_LOSO_mean_within_study_AUC |   study_R2_condition_controlled |   study_prediction_balanced_accuracy |
|:---------------------------------------------------|--------------------------------:|---------------------------------:|-------------------------------------:|--------------------------------:|-------------------------------------:|
| raw                                                |                      0.00787524 |                         0.615319 |                             0.709835 |                       0.0785634 |                             0.75637  |
| mmuphin                                            |                      0.00884871 |                         0.623513 |                             0.687841 |                       0.0300456 |                             0.673665 |
| mmguide_resgrl_lam02_anchor01_pres2_cov005_delta05 |                      0.00807136 |                         0.625753 |                             0.701762 |                       0.0570487 |                             0.653568 |
| mmguide_resgrl_lam05_anchor01_pres2_cov005_delta05 |                      0.00716986 |                         0.637417 |                             0.695249 |                       0.0637028 |                             0.666519 |
| mmguide_resgrl_lam02_anchor02_pres5_cov01_delta1   |                      0.00794755 |                         0.604553 |                             0.710332 |                       0.0704798 |                             0.65528  |
| mmguide_resgrl_lam05_noanchor_pres5_cov01_delta1   |                      0.00794199 |                         0.617242 |                             0.711925 |                       0.0621518 |                             0.661429 |
| mmguide_resgrl_lam02_anchor1_pres2_cov005_delta05  |                      0.00826884 |                         0.643107 |                             0.708961 |                       0.0579232 |                             0.623862 |
| mmguide_resgrl_lam02_anchor2_pres1_cov005_delta1   |                      0.0087812  |                         0.637259 |                             0.719114 |                       0.0569993 |                             0.668816 |
| mmguide_resgrl_lam05_anchor1_pres1_cov005_delta1   |                      0.00815771 |                         0.638647 |                             0.697    |                       0.0826019 |                             0.707425 |

## Geometry Diagnostics

| method                                             |   effective_rank |   pc1_variance |   pc2_variance |   mean_delta_l2 |   mean_delta_mse |
|:---------------------------------------------------|-----------------:|---------------:|---------------:|----------------:|-----------------:|
| raw_log_std                                        |          364.099 |      0.0327213 |      0.0293304 |         0       |       0          |
| mmuphin_log_std                                    |          360.872 |      0.0314922 |      0.028415  |         0       |       0          |
| mmguide_resgrl_lam02_anchor01_pres2_cov005_delta05 |          361.566 |      0.0465439 |      0.0294226 |         6.71107 |       0.102551   |
| mmguide_resgrl_lam05_anchor01_pres2_cov005_delta05 |          360.671 |      0.0467684 |      0.0321441 |         8.45533 |       0.152195   |
| mmguide_resgrl_lam02_anchor02_pres5_cov01_delta1   |          364.397 |      0.0320075 |      0.0298061 |         1.03784 |       0.00416087 |
| mmguide_resgrl_lam05_noanchor_pres5_cov01_delta1   |          363.919 |      0.0331203 |      0.028671  |         3.6315  |       0.0344381  |
| mmguide_resgrl_lam02_anchor1_pres2_cov005_delta05  |          364.004 |      0.0359564 |      0.0312588 |         4.13146 |       0.0468648  |
| mmguide_resgrl_lam02_anchor2_pres1_cov005_delta1   |          359.758 |      0.0786709 |      0.0288207 |         7.43024 |       0.133257   |
| mmguide_resgrl_lam05_anchor1_pres1_cov005_delta1   |          354.207 |      0.0884852 |      0.0468086 |        11.7171  |       0.293086   |

## Reading

- The residual design avoids the obvious low-dimensional collapse seen in old abundance GRL: effective rank stays close to raw/MMUPHin in log-standardized space for the conservative configurations.
- It does not beat MMUPHin on Study R2. The best Study R2 among this sweep remains around 0.057, while MMUPHin is 0.030.
- Several residual configurations preserve disease LOSO AUC better than MMUPHin and lower Study BA slightly below MMUPHin, but Study R2 remains too high.
- Stronger MMUPHin anchor does not solve Study R2 here; it can inflate PC1 variance when the correction is too unconstrained.
- This is a cleaner negative/partial result than old cw0.1: safer geometry, but insufficient batch-effect removal under the frozen full551 R evaluator.
