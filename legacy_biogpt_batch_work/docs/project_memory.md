# Ali Lab Shared Project Memory

Last updated: 2026-05-19

## Purpose

This folder is the shared memory layer for the local `scgpt` and `biogpt/dataset_v3` work. The two projects should remain physically separate, but they share research concepts, paths, terminology, and professor-facing outputs through this folder.

## Project Map

- scGPT source project:
  `C:\Users\Yuanshu\Desktop\Ali lab\scgpt\model\scGPT-main`
- BiomeGPT/dataset_v3 project:
  `C:\Users\Yuanshu\Desktop\Ali lab\biogpt\dataset_v3`
- BiomeGPT full Colab notebook:
  `C:\Users\Yuanshu\Desktop\Ali lab\biogpt\dataset_v3\BiomeGPT_full_pipeline_colab.ipynb`
- Species taxonomy file:
  `C:\Users\Yuanshu\Desktop\Ali lab\biogpt\dataset_v3\species_taxonomy_filled_validated_Serena.xlsx`
- External validation data:
  `C:\Users\Yuanshu\Desktop\Ali lab\biogpt\dataset_v3\ExVal`

## Shared Scientific Analogy

- scGPT gene embedding maps to BiomeGPT species prompt.
- scGPT cell embedding maps to BiomeGPT sample prompt.
- BiomeGPT species embedding should be taxonomy-aware:
  `Domain + Kingdom + Phylum + Class + Order + Family + Genus + Species`.
- Taxonomy is sample-independent prior biological knowledge.

## Current BiomeGPT Pipeline Status

The active working notebook is `BiomeGPT_full_pipeline_colab.ipynb`. It contains the model implementation and pipeline cells directly inside the notebook for Colab use.

Smoke-test evidence exists locally, but it is not scientific performance evidence. The final full runs should be performed later on Colab.

## Local Compute Rule

Use the local machine/GPU only for:
- smoke tests
- reduced debugging runs
- validation checks
- notebook refactoring

Do not run full pretraining or full-scale fine-tuning locally unless explicitly requested.

## Colab Path Contract

The notebook expects this Google Drive layout:

```text
/content/drive/MyDrive/dataset_v3
/content/drive/MyDrive/dataset_v3/ExVal
/content/drive/MyDrive/dataset_v3/species_taxonomy_filled_validated_Serena.xlsx
```

## Professor-Facing Artifacts

The pipeline should produce:

- `pipeline_status.md`
- `bugs_fixed.md`
- `results_summary.md`
- `next_steps.md`
- `data_contract_summary.json`
- `species_alignment_summary.csv`
- `taxonomy_completeness_summary.csv`
- `label_balance_summary.csv`
- `species_prompt_taxonomy_neighbor_purity.json`
- `sample_prompt_gut_vs_nongut_summary.json`
- `exval_hd_metrics.json`
- `exval_confusion_matrix.csv`
- `exval_metric_sanity_checks.json`
- `exval_probability_histogram.png`

## Important Interpretation Rule

Always distinguish:

- smoke-test evidence: pipeline executes and diagnostics are produced
- scientific evidence: full Colab run with complete training settings
- unresolved uncertainty: any suspicious probability collapse, one-class predictions, or poor H/D balance

