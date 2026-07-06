# BiomeGPT Batch Correction Reusable Package

This is a lightweight reusable package for continuing the real-study-id batch correction work.

## Included

- Core Python scripts in `dataset_v3/*.py`
- Real study-id annotation CSV and summaries
- Reports in `reports/`
- Key diagnostic/status CSV/JSON/MD files
- Saved corrected embeddings:
  - `dataset_v3/outputs_real_study_embedding_correction_saved/real_study_conservative_safe_mean_center_corrected_embeddings.npz`
  - `dataset_v3/outputs_real_study_embedding_correction_saved/real_study_high_only_mean_center_corrected_embeddings.npz`
- Uploaded study-id mapping:
  - `BiomeGPT_species_samples_studyIDs.csv`

## Not Included

To keep the package small, this excludes:

- raw large uploaded zip archives
- full intermediate output directories
- model checkpoints
- repeated embedding caches
- LaTeX aux/log/out files

## Most Important Report

- `reports/batch_correction_full_workflow_report.md`
- `reports/batch_diagnostics_report.pdf`

## Current Best Usable Result

The strongest practical corrected representation is:

```text
dataset_v3/outputs_real_study_embedding_correction_saved/real_study_conservative_safe_mean_center_corrected_embeddings.npz
```

It is based on cross-fitted real-study mean centering using conservative-safe studies.

## To Resume Full Training

You will need to restore or re-download the original large abundance files and checkpoints, especially:

```text
dataset_v3/abund_pretraining_phase2_gut.csv.zip
dataset_v3/meta_pretraining_phase2_gut.csv
dataset_v3/outputs_taxonomy_notebook/taxonomy_checkpoint_stage2.pt
```

Those are intentionally not included in this lightweight package.

