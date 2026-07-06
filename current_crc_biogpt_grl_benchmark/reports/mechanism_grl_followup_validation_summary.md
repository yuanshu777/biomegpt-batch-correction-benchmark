# Mechanism-Only GRL Follow-Up Validation

## Scope

This follow-up checks the previous mechanism-only best candidate:

`grl_mech_context_only_l8_lam10_rw5_rel1_var1`

The checks were:

1. Condition amplification diagnostics on the full 551-sample CRC benchmark.
2. Strict LOSO correction, where each held-out study is transformed by a model trained only on the other studies.
3. CRC389 overlap run so the abundance method can be compared with BiomeGPT CLS baselines on the same samples.

## 1. Condition Amplification Diagnostic

| method | effective rank | PC1 condition AUC | condition R2 linear | study R2 linear |
|---|---:|---:|---:|---:|
| raw | 221.731 | 0.626 | 0.0059 | 0.0357 |
| MMUPHin | 224.303 | 0.522 | 0.0063 | 0.0224 |
| old GRL cw0.1 | 10.393 | 0.534 | 0.0230 | 0.0583 |
| old GRL cw0.01 | 10.103 | 0.630 | 0.0184 | 0.0619 |
| mechanism-only best | 24.255 | 0.587 | 0.0132 | 0.0126 |

Mechanism-only best is better than old GRL on collapse diagnostics: effective rank rises from about 10 to 24, and condition R2 linear is lower than old `cw0.1`. It is still much lower-rank than raw/MMUPHin, so it is not fully structure-preserving.

## 2. Strict LOSO Correction

Python LOSO classifier results:

| method | mean within-study AUC | overall AUC | balanced accuracy |
|---|---:|---:|---:|
| raw Python LOSO classifier | 0.744 | 0.760 | 0.692 |
| MMUPHin Python LOSO classifier | 0.750 | 0.770 | 0.703 |
| LOSO mechanism-only GRL | 0.613 | 0.609 | 0.593 |

OOF abundance matrix evaluated by the original R benchmark:

| metric | estimate |
|---|---:|
| Study R2, condition-controlled | 0.059956 |
| Study prediction balanced accuracy | 0.932084 |
| Disease LOSO mean within-study AUC | 0.590779 |
| Condition R2, study-controlled | 0.011552 |

This is the most important result: the full-data mechanism-only candidate does not survive strict LOSO/cross-fitted correction. It does not preserve disease signal and the OOF study classifier becomes highly predictive.

## 3. CRC389 Overlap

| method | Study R2 | Study BA | Disease LOSO AUC | Condition R2 |
|---|---:|---:|---:|---:|
| Raw abundance | 0.0272 | 0.795 | 0.734 | 0.0071 |
| MMUPHin adjusted abundance | 0.0160 | 0.569 | 0.776 | 0.0073 |
| Mechanism-only GRL abundance | 0.0170 | 0.382 | 0.686 | 0.0176 |
| BiomeGPT raw CLS | 0.0572 | 0.712 | 0.686 | 0.0093 |
| BiomeGPT study-mean-centered CLS | 0.0007 | 0.049 | 0.694 | 0.0099 |

On the same 389 samples, mechanism-only GRL abundance reduces study BA strongly and has study R2 close to MMUPHin, but disease LOSO AUC remains below MMUPHin and condition R2 is elevated. BiomeGPT mean-centering is still a very strong simple CLS baseline.

## Current Conclusion

Do not claim mechanism-only GRL beats MMUPHin.

The full-data result was promising, but strict LOSO correction shows poor held-out-study generalization. The likely issue is that the full-data decoder learns a dataset-specific transformation that does not transfer cleanly to unseen studies.

## What This Means For Next Work

The next real improvement should focus on inductive stability, not just full-data benchmark scores:

- reduce decoder capacity further or regularize harder
- train with study-held-out validation during correction training
- select checkpoints by held-out-study preservation, not full-data metrics
- keep BiomeGPT CLS mean-centering as a baseline that GRL must beat

## Output Files

- amplification diagnostics: `outputs\metrics\condition_amplification_full_crc_diagnostics.csv`
- strict LOSO summary: `outputs\metrics\grl_abundance_loso_correction_summary.csv`
- strict LOSO per-fold: `outputs\metrics\grl_abundance_loso_correction_per_fold_metrics.csv`
- OOF R evaluator report: `C:\Users\Yuanshu\Documents\new_attemp_batch\crc_controlled_benchmark\reports\methods\loso_grl_mech_context_only_l8_lam10_rw5_rel1_var1_oof`
- CRC389 table: `outputs\metrics\mechanism_grl_crc389_primary_table.csv`
- CRC389 report: `reports\mechanism_grl_crc389_summary.md`
