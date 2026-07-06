# Lab Handoff README: BiomeGPT / MMUPHin CRC Batch-Correction Work

Prepared: 2026-07-06  
Workspace root: `C:\Users\Yuanshu\Documents\new_attemp_batch`

This file is a handoff index for the code, reports, benchmark outputs, and model artifacts generated during the BiomeGPT/MMUPHin CRC batch-correction exploration. It is written to make the project understandable without relying on prior chat history.

## 1. High-Level Project Goal

The project tested whether a BiomeGPT/scGPT-style representation-correction method can reduce study/cohort effects in microbiome data while preserving disease signal.

The controlled benchmark used for most method development is the MMUPHin CRC/control dataset:

- 551 samples
- 5 studies
- 484 species-level abundance features
- CRC/control disease setting
- measurable study/batch effect in raw abundance
- MMUPHin-adjusted abundance available as a reference correction

The main scientific question was:

> Can a foundation-model-style microbiome representation reduce study/batch signal while preserving CRC/control signal, under the same evaluation logic used for MMUPHin?

## 2. Most Important Current Conclusion

There is a real and useful project here, but the current results should be framed conservatively.

The strongest defensible conclusion is:

> BiomeGPT CLS embeddings and raw abundance both contain measurable study/cohort signal. Simple post-hoc GRL correction can reduce study classifier predictability, but it does not reliably remove global study-associated variance. More promising directions are architecture-level correction methods, such as invariant/nuisance splitting, study-conditioned reconstruction, conditional CORAL/MMD, and adapter/LoRA-style BiomeGPT post-training.

Do not claim that the current GRL method definitively beats MMUPHin.

## 3. Recommended Files/Folders To Share

Recommended main upload set:

1. `crc_biogpt_grl_benchmark\`
   - Main reusable package for the CRC/BiomeGPT/GRL benchmark work.
   - Includes `src`, `scripts`, `reports`, `outputs`, and legacy package inventory.

2. `crc_controlled_benchmark\`
   - Canonical MMUPHin-style controlled benchmark folder.
   - Includes raw/MMUPHin reports, R evaluator outputs, PCA plots, and method-specific comparison reports.

3. `crc_overlap_check\`
   - Overlap audit between MMUPHin CRC samples/studies and prior BiomeGPT data.

4. `outputs_mmuphin_dataset_scouting\`
   - Initial MMUPHin CRC dataset scouting output.

5. `output\pdf\full551_progress_report\crc_batch_correction_progress_report.pdf`
   - Professor-facing summary PDF for the full 551-sample progress report.

6. `output\pdf\full551_progress_report\crc_batch_correction_progress_report.tex`
   - LaTeX source for the above PDF.

7. `output\pdf\mmuphin_crc_professor_report.pdf`
   - Earlier professor-facing report on the MMUPHin CRC benchmark.

8. Top-level R/Python scripts:
   - `prepare_crc_controlled_benchmark.R`
   - `evaluate_crc_method.R`
   - `crc_benchmark_utils.R`
   - `validate_crc_benchmark.R`
   - `mmuphin_crc_scouting.R`
   - `crc_overlap_check.py`
   - `plot_old_cw01_mmuphin_pca.R`
   - `plot_mechanism_best_mmuphin_pca.R`
   - `plot_best_residual_grl_mmuphin_pca.R`

9. Checkpoints, only if allowed and needed:
   - `taxonomy_checkpoint_stage1 (1).pt`
   - `taxonomy_checkpoint_stage2 (1).pt`

Large checkpoints should be uploaded separately if the sharing system has file-size limits.

## 4. Files/Folders To Avoid Uploading Unless Needed

Avoid uploading these unless there is a specific reason:

- `.git\`
- `.codex\`
- `.agents\`
- `tmp\`
- Python `__pycache__` folders
- LaTeX intermediate files: `*.aux`, `*.log`, `*.out`, `*.toc`
- rendered PDF page images under `output\pdf\full551_progress_report\rendered\`
- duplicate or stale zip files unless the recipient specifically wants them

Existing zip files:

- `crc_biogpt_grl_benchmark.zip`
  - Useful as a snapshot, but may not include the very latest top-level handoff file.

- `biomegpt_reusable_20260521_batch_correction.zip`
  - Legacy source package from earlier experiments; useful for historical context, but not the clean current benchmark package.

## 5. Key Reports To Read First

Start with these reports in this order:

1. `output\pdf\full551_progress_report\crc_batch_correction_progress_report.pdf`
   - Best high-level summary.

2. `crc_biogpt_grl_benchmark\reports\full551_benchmark_reproduction_summary.md`
   - Reproduces the MMUPHin full 551-sample benchmark.

3. `crc_biogpt_grl_benchmark\reports\full551_grl_abundance_summary.md`
   - Abundance-level GRL experiments on the full CRC benchmark.

4. `crc_biogpt_grl_benchmark\reports\mmuphin_guided_residual_grl_full551_summary.md`
   - Residual GRL method designed to avoid geometry collapse.

5. `crc_biogpt_grl_benchmark\reports\full551_biogpt_cls_extraction_audit.md`
   - Audit of BiomeGPT CLS extraction on all 551 MMUPHin CRC samples.

6. `crc_biogpt_grl_benchmark\reports\full551_biogpt_cls_summary.md`
   - Raw and corrected BiomeGPT CLS results on 551 samples.

7. `crc_biogpt_grl_benchmark\reports\crc389_overlap_audit_vs_full551.md`
   - Explains why 389 overlap is exploratory and 551 is canonical.

8. `crc_overlap_check\crc_overlap_check_summary.md`
   - Sample/study/disease overlap between prior BiomeGPT data and MMUPHin CRC.

## 6. Canonical Benchmark Results

The original MMUPHin-style controlled benchmark is the 551-sample CRC dataset. The key metrics are:

| Method | Study R2, condition-controlled | Study BA | Disease LOSO AUC | Condition R2, study-controlled | Interpretation |
|---|---:|---:|---:|---:|---|
| Raw abundance | 0.0786 | 0.756 | 0.710 | 0.0079 | Strong study signal, disease signal present |
| MMUPHin adjusted | 0.0300 | 0.674 | 0.688 | 0.0088 | Strong study-R2 reduction, disease mostly preserved |
| Old GRL cw0.1 | 0.0328 | 0.553 | 0.882 | 0.0723 | Suspicious disease amplification / geometry compression |
| Mechanism-only full-data GRL | 0.0114 | 0.378 | 0.688 | 0.0233 | Looks strong transductively, but fails stricter LOSO/OOF validation |
| Residual GRL best tradeoff | about 0.057 | about 0.654 | about 0.702 | about 0.008 | Healthier geometry, but does not beat MMUPHin on Study R2 |

Important interpretation:

- MMUPHin remains the strongest reliable baseline for Study R2 reduction.
- Old GRL cw0.1 should not be claimed as a clean win because its Disease AUC is likely inflated by condition-axis amplification.
- Mechanism-only full-data GRL should not be claimed as a final result because strict LOSO/OOF validation did not hold.
- Residual GRL is scientifically cleaner, but it still needs stronger Study R2 reduction.

## 7. BiomeGPT CLS Results

BiomeGPT CLS extraction on the full 551-sample MMUPHin CRC benchmark succeeded technically:

- 551 CLS embeddings extracted
- 512-dimensional CLS representation
- 357 / 484 MMUPHin species mapped to the checkpoint vocabulary
- mapped fraction: about 73.8%

The current full551 raw CLS metrics are:

| Representation | Study R2 | Study BA | Disease LOSO AUC | Condition R2 |
|---|---:|---:|---:|---:|
| BiomeGPT raw CLS 551 | 0.1578 | 0.684 | 0.694 | 0.0110 |
| Mean-centered CLS 551 | 0.0002 | 0.099 | 0.696 | not primary |
| NoMean conditional GRL CLS 551 | 0.1489 | 0.533 | 0.691 | 0.0112 |
| Split conditional CORAL CLS 551 | 0.0842 | 0.587 | 0.675 | 0.0183 |
| Old cw0.1-style CLS 551 | 0.0991 | 0.460 | 0.655 | 0.0153 |

Interpretation:

- Raw BiomeGPT CLS contains substantial study signal.
- Simple GRL reduces study classifier predictability but does not sufficiently reduce Study R2.
- Split/CORAL-style architectures reduce Study R2 more strongly, but disease signal and geometry need further validation.
- Current CLS results are useful but should be treated as preliminary because species coverage is incomplete.

## 8. CRC389 Overlap Results

The 389-sample overlap set is useful for connecting MMUPHin CRC samples to prior BiomeGPT data, but it is not the canonical benchmark.

Key BiomeGPT raw CLS 389 metrics:

| Representation | Study R2 | Study BA | Disease LOSO AUC | Condition R2 |
|---|---:|---:|---:|---:|
| BiomeGPT raw CLS 389 | 0.0572 | 0.7119 | 0.6864 | 0.0093 |

Some GRL/adapter methods reduced Study BA substantially but only mildly reduced Study R2. The most promising architecture-level 389 result was an invariant/nuisance split plus conditional CORAL-style method, which reduced Study R2 more clearly, but this needs full551 and LOSO/cross-fitted validation before claiming success.

## 9. Main Code Entry Points

Main package:

- `crc_biogpt_grl_benchmark\src\biogpt_core\`
  - BiomeGPT checkpoint loading, model definition, data conversion, and CLS extraction.

- `crc_biogpt_grl_benchmark\src\grl_correction\`
  - GRL correction modules, adversaries, losses, cross-fitting, NoMean adapter, split adapter.

- `crc_biogpt_grl_benchmark\src\evaluation\`
  - Python evaluation helpers, probes, PCA plots, partial R2, and PERMANOVA-like utilities.

- `crc_biogpt_grl_benchmark\src\mmuphin_bridge\`
  - MMUPHin CRC loading and overlap utilities.

Important scripts:

- `crc_biogpt_grl_benchmark\scripts\build_full551_canonical_outputs.py`
  - Collects canonical full551 outputs and summary artifacts.

- `crc_biogpt_grl_benchmark\scripts\extract_biogpt_cls_for_crc_overlap.py`
  - Extracts BiomeGPT CLS for CRC overlap samples.

- `crc_biogpt_grl_benchmark\scripts\run_full551_biogpt_cls_corrections.py`
  - Runs full551 BiomeGPT CLS correction experiments.

- `crc_biogpt_grl_benchmark\scripts\run_full551_biogpt_old_cw01_grl.py`
  - Runs old cw0.1-style GRL on BiomeGPT CLS for comparison.

- `crc_biogpt_grl_benchmark\scripts\train_mmuphin_guided_residual_grl_full551.py`
  - Residual GRL full551 method with MMUPHin-inspired constraints.

- `crc_biogpt_grl_benchmark\scripts\run_cls_architecture_ablation_crc389.py`
  - 389-sample CLS architecture ablation.

- `crc_biogpt_grl_benchmark\scripts\run_grl_crossfit_crc389.py`
  - Cross-fitted GRL evaluation for 389 abundance benchmark.

- `crc_biogpt_grl_benchmark\scripts\diagnose_condition_amplification_full_crc.py`
  - Diagnostics for condition amplification and low-dimensional collapse.

R evaluator scripts:

- `prepare_crc_controlled_benchmark.R`
- `evaluate_crc_method.R`
- `crc_benchmark_utils.R`
- `validate_crc_benchmark.R`

## 10. Important Figures

Professor-facing figures:

- `output\pdf\full551_progress_report\figures\full551_raw_mmuphin_pca.png`
- `output\pdf\full551_progress_report\figures\full551_old_grl_pca.png`
- `output\pdf\full551_progress_report\figures\full551_residual_grl_pca.png`

Canonical MMUPHin PCA plots:

- `crc_controlled_benchmark\reports\plots\raw_pca_by_study.png`
- `crc_controlled_benchmark\reports\plots\raw_pca_by_condition.png`
- `crc_controlled_benchmark\reports\plots\mmuphin_pca_by_study.png`
- `crc_controlled_benchmark\reports\plots\mmuphin_pca_by_condition.png`

Old GRL cw0.1 PCA:

- `crc_controlled_benchmark\reports\methods\grl_abundance_l8_lam10_cw01_rw1\plots\raw_mmuphin_old_cw01_pca_by_study_panel.png`
- `crc_controlled_benchmark\reports\methods\grl_abundance_l8_lam10_cw01_rw1\plots\raw_mmuphin_old_cw01_pca_by_condition_panel.png`

Best residual GRL PCA:

- `crc_controlled_benchmark\reports\methods\mmguide_resgrl_lam02_anchor01_pres2_cov005_delta05\plots\raw_mmuphin_best_residual_grl_pca_by_study_panel.png`
- `crc_controlled_benchmark\reports\methods\mmguide_resgrl_lam02_anchor01_pres2_cov005_delta05\plots\raw_mmuphin_best_residual_grl_pca_by_condition_panel.png`

## 11. What Not To Overclaim

Please do not claim:

- that GRL definitively beats MMUPHin;
- that old cw0.1 is a clean biological improvement;
- that 389 overlap is the primary benchmark;
- that BiomeGPT CLS correction is final;
- that current results are a full foundation-model-level batch correction method.

Safer language:

- "preliminary evidence"
- "controlled benchmark scaffold"
- "post-hoc representation correction prototype"
- "suggests a direction for batch-aware BiomeGPT adapter/LoRA post-training"
- "requires multi-disease validation and stricter LOSO/cross-fitted correction"

## 12. Recommended Next Scientific Step

The next strong direction is not to keep tuning vanilla GRL. The better next method is:

> Batch-aware BiomeGPT adapter/LoRA post-training with invariant/nuisance splitting, study-conditioned reconstruction, residual conditional adversary, conditional CORAL/MMD, and anti-collapse constraints.

Proposed architecture:

```text
BiomeGPT backbone
        |
 adapter / LoRA
        |
 z_inv       z_batch / nuisance
        |          |
 final CLS    decoder side-channel
```

Training objectives:

- masked abundance/species reconstruction;
- study-conditioned decoder reconstruction;
- residual conditional study adversary;
- conditional CORAL or MMD within disease/control strata;
- geometry / variance / effective-rank preservation;
- no strong condition classifier unless carefully validated.

Evaluation requirements:

- full551 MMUPHin CRC benchmark;
- CRC389 overlap only as exploratory connection to older BiomeGPT data;
- at least one or two additional multi-study disease benchmarks such as IBD, T2D/metabolic, RA, or PD;
- LOSO or cross-fitted correction to avoid transductive leakage;
- compare against raw abundance, MMUPHin, raw BiomeGPT CLS, post-hoc GRL CLS, and batch-aware BiomeGPT CLS.

## 13. Suggested Short Message To Lab/Professor

The project produced a cleaned MMUPHin CRC benchmark and a reusable BiomeGPT/GRL benchmark scaffold. The main reliable result is that MMUPHin remains the strongest controlled baseline for study-R2 reduction, while BiomeGPT CLS embeddings contain measurable study signal that can be partially reduced by representation correction. Simple GRL reduces study classifier accuracy but does not reliably remove global study-associated variance. More promising results come from architecture-level correction ideas, especially residual correction, invariant/nuisance splitting, conditional CORAL/MMD, and study-conditioned reconstruction. The next step should be a true batch-aware BiomeGPT adapter/LoRA post-training method evaluated with LOSO/cross-fitted correction and additional multi-study disease benchmarks.

## 14. Upload Checklist

Before uploading:

- Include this file at the root of the shared folder.
- Include `crc_biogpt_grl_benchmark\`.
- Include `crc_controlled_benchmark\`.
- Include `crc_overlap_check\`.
- Include professor-facing PDFs under `output\pdf\`.
- Include checkpoints only if the recipient needs rerunnable CLS extraction.
- Exclude `.codex`, `.agents`, `.git`, `tmp`, `__pycache__`, and LaTeX intermediates.
- If uploading to GitHub, consider using Git LFS or external storage for `.pt` checkpoints and large matrices.

## 15. Quick Integrity Note

A quick keyword scan did not identify obvious plaintext credentials in the main code/report folders. Some false positives occur because the project naturally contains words like "token" for microbiome/tokenization/sample IDs and because binary PDFs can match arbitrary strings. A final manual review is still recommended before public upload.

