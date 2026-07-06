# Focused Package Repair Summary

## 1. What Old Code Was Found?

The old reusable package `biomegpt_reusable_20260521_batch_correction.zip` was
found in the project root and copied into `legacy_source/`. It contains legacy
BiomeGPT batch-effect scripts, metadata annotations, diagnostics, reports, and
saved mean-centering embedding outputs. The detailed inventory is in
`reports/old_package_inventory.md` and `reports/old_package_inventory.csv`.

Useful source-material areas include:

- batch-adversarial correction
- batch-token pretraining
- study-conditioned decoder correction
- centroid/mean-centering/distillation correction
- batch diagnostics and probe metrics
- BiomeGPT taxonomy / abundance pipeline code

The old package is not treated as current or fully runnable.

## 2. What BiomeGPT Core Pieces Were Recovered?

The focused package now has a minimal BiomeGPT core scaffold:

- `src/biogpt_core/model.py`
- `src/biogpt_core/data.py`
- `src/biogpt_core/checkpoint.py`
- `src/biogpt_core/extract_cls.py`
- `src/biogpt_core/notes.md`

It supports loading the configured taxonomy-aware checkpoint and can subset an
existing raw CLS matrix to the CRC overlap samples. After switching to the local
Miniconda Python environment with PyTorch, `scripts/extract_biogpt_cls_for_crc_overlap.py`
successfully generated `outputs/crc_overlap_benchmark/biogpt_raw_cls_389.csv`
with 389 samples and 512 CLS dimensions.

## 3. What GRL/scGPT-style Pieces Were Recovered or Rebuilt?

The GRL/scGPT-style correction module was rebuilt as a small embedding-level
training scaffold:

- `src/grl_correction/grl.py`
- `src/grl_correction/adversary.py`
- `src/grl_correction/conditioned_decoder.py`
- `src/grl_correction/losses.py`
- `src/grl_correction/train_grl.py`
- `src/grl_correction/notes.md`

The local minimum version trains on sample x embedding matrices with study and
CRC/control metadata. It includes a study adversary through gradient reversal
and a condition head for disease-signal monitoring. It does not pretrain
BiomeGPT from scratch.

## 4. What MMUPHin Bridge Pieces Exist?

The package includes:

- `src/mmuphin_bridge/load_crc.py`
- `src/mmuphin_bridge/build_overlap.py`
- `src/mmuphin_bridge/run_mmuphin_subset.R`
- `src/mmuphin_bridge/notes.md`
- `scripts/build_crc_overlap_benchmark.py`
- `scripts/run_mmuphin_on_crc_overlap.R`

The builder uses the existing overlap audit and MMUPHin benchmark files to
create the 389-sample direct-comparison assets.

`scripts/run_mmuphin_on_crc_overlap.R` was also run successfully. The main
`outputs/crc_overlap_benchmark/mmuphin_adjusted_abundance_389.csv` now comes
from MMUPHin rerun on the 389-sample overlap subset. The earlier full-551
adjusted matrix subset is retained as
`outputs/crc_overlap_benchmark/mmuphin_adjusted_abundance_389_from_full551.csv`
for auditability.

## 5. What Files Are Missing?

Expected missing or not-current assets:

- GRL-corrected CLS output, until training is run later

The working directory now contains `taxonomy_checkpoint_stage1 (1).pt` and
`taxonomy_checkpoint_stage2 (1).pt`; stage2 is configured as the default
checkpoint. `reports/checkpoint_inventory.md` records the checkpoint sizes and
PyTorch zip structure. The extractor is wired to the legacy taxonomy-aware model
definition and can attempt checkpoint-based CLS extraction in an environment
with PyTorch installed.

## 6. What Can Run Locally Now?

These local steps can run without GPU:

- build 389-sample overlap manifest and abundance matrices
- rerun MMUPHin on the 389 overlap subset
- inspect old package inventory
- run asset checks
- run synthetic GRL smoke tests
- evaluate any available sample x feature/embedding method matrix

In the bundled Python runtime, `pytest`, `scikit-learn`, and `torch` were not
available. The local Miniconda Python environment does have PyTorch and was used
to extract raw CLS embeddings from the stage2 checkpoint. Evaluation also ran
there for raw abundance, MMUPHin-adjusted abundance, and raw BiomeGPT CLS;
GRL-corrected CLS remains missing until training is run later.

## 7. What Requires A100 Later?

A100 or equivalent GPU is only needed if we later run full BiomeGPT inference
or representation training at scale. The current package does not require A100
for setup, tests, or abundance/MMUPHin benchmarking.

## 8. Next Scientific Experiment

Primary benchmark:

- Use the 389 overlapping samples.
- Compare raw abundance, MMUPHin-adjusted abundance, raw BiomeGPT CLS, and
  GRL/scGPT-style corrected BiomeGPT CLS.
- Report study predictability, study macro-F1, CRC/control AUROC, CRC/control
  balanced accuracy, CRC/control macro-F1, and PCA diagnostics.

Reference benchmark:

- Use the full 551 MMUPHin CRC samples for raw abundance versus MMUPHin-adjusted
  abundance only.

No success claim is made yet. This package is for repair, asset alignment, and
benchmark setup.
