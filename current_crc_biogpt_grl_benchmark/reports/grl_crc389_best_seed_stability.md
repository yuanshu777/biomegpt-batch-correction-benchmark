# GRL CRC389 Best Setting Seed Stability

## Scope

This repeats the best local abundance-level GRL setting from the tuning sweep across five random seeds. It is still a local scaffold check, not a final BiomeGPT/scGPT result.

## Best Setting Repeated

- `latent_dim=8`
- `lambda_grl=10.0`
- `lambda_schedule=linear`
- `warmup_fraction=0.1`
- `condition_weight=0.1`
- `preserve_weight=0.001`
- `condition_aware_adversary=True`

## Raw Reference

- Raw study balanced accuracy: 0.621
- Raw CRC/control AUROC: 0.713

## Seed Results

|   seed |   study_balanced_accuracy |   condition_auc |   condition_balanced_accuracy |   condition_macro_f1 |
|-------:|--------------------------:|----------------:|------------------------------:|---------------------:|
|      7 |                  0.514101 |        0.701148 |                      0.649291 |             0.629717 |
|     42 |                  0.453284 |        0.802158 |                      0.740308 |             0.721988 |
|     99 |                  0.491497 |        0.797199 |                      0.723132 |             0.700853 |
|    123 |                  0.582015 |        0.641822 |                      0.577718 |             0.557688 |
|   2026 |                  0.496756 |        0.701689 |                      0.649772 |             0.623923 |

## Aggregate

|      |   study_balanced_accuracy |   condition_auc |   condition_balanced_accuracy |
|:-----|--------------------------:|----------------:|------------------------------:|
| mean |                 0.507531  |       0.728803  |                     0.668044  |
| std  |                 0.0471886 |       0.0691457 |                     0.0653855 |
| min  |                 0.453284  |       0.641822  |                     0.577718  |
| max  |                 0.582015  |       0.802158  |                     0.740308  |

## Interpretation

- Across all five seeds, study balanced accuracy stayed below raw abundance, so the bottlenecked GRL direction is not just a single-seed accident.
- CRC/control AUROC is less stable. Some seeds preserve or improve disease signal, while others fall below raw abundance.
- The next useful adjustment is stability-oriented: repeat-seed selection, early stopping on external probes, or a small validation split for choosing the checkpoint.

## Output Files

- `seed_stability_metrics`: `outputs\metrics\grl_crc389_best_seed_stability.csv`
