# CODEX Project Context: BiomeGPT + scGPT Batch Correction

## Role Of This Package

This package is for VS Code SSH + Codex. It contains the files needed to continue the BiomeGPT/dataset_v3 project and enough context for a new Codex session to understand the scientific goals.

## Active Notebook

Use:

```text
dataset_v3/BiomeGPT_full_pipeline_vscode_ssh.ipynb
```

The original Colab notebook is also included:

```text
dataset_v3/BiomeGPT_full_pipeline_colab.ipynb
```

## Hard Constraint

Do not run full-scale pretraining or fine-tuning by default. Keep `SMOKE_TEST=True` and `FULL_RUN=False` unless Yuanshu explicitly asks for a full run.

## BiomeGPT Scientific Goals

1. Reproduce a BiomeGPT-style species abundance foundation model.
2. Phase 1 pretraining: gut + non-gut, 30 epochs, 32 bins.
3. Phase 2 domain adaptation: gut only, 3-5 epochs.
4. Implement taxonomy-aware species embedding:
   Domain + Kingdom + Phylum + Class + Order + Family + Genus + Species.
5. Extract prompts:
   - Species prompt = taxonomy-aware species embedding.
   - Sample prompt = sample-level CLS embedding.
6. Representation analysis:
   - UMAP gut vs non-gut sample prompts.
   - UMAP species prompts colored by genus/family/order/phylum.
   - Quantitative kNN purity / silhouette / logistic probe metrics.
7. Fine-tune Healthy vs Diseased classifier using `_prev3` files and ExVal.
8. Report accuracy, F1, AUROC, macro-accuracy, macro-F1, macro-AUROC, H-accuracy, D-accuracy.

## Existing Important Fixes

- Masked abundance prediction should use MSE, not CE over class bins.
- Attention mask should follow BiomeGPT logic:
  - zero-abundance species do not participate in attention,
  - unmasked non-zero species can see each other,
  - masked species can see unmasked species and self,
  - loss only on masked non-zero species.
- ExVal species columns must align exactly to training species vocabulary; missing species become zero.
- Synthetic diseased augmentation must preserve original zeros as zeros, clip negatives to zero, and use std=5.

## Batch-Correction Direction From scGPT

scGPT batch integration uses:

- `batch_id` labels from metadata.
- Domain-specific batch normalization option (`DSBN`).
- Domain adversarial branch (`AdversarialDiscriminator`) with gradient reversal.
- DAB loss (`criterion_dab`) added to main training loss.

BiomeGPT adaptation idea:

- Use sample prompt / CLS embedding as the representation for batch discrimination.
- Add a small batch MLP with gradient reversal to make sample embeddings less batch-predictive.
- Before correction, run batch probes to verify whether batch is encoded.
- After correction, verify batch probe performance decreases while phenotype/H-D probe performance remains stable.

## Batch Annotation Data

External batch labels are in:

```text
dataset_v3/meta_pretraining_phase2_gut_batch_annotation_external_enriched.csv
```

Key columns:

- `batch_label_external_recommended`
- `external_confidence`
- `external_source`
- `external_study_accession`
- `phenotype_confounding_warning`
- `needs_manual_review`
- `safe_for_final_batch_correction_conservative`

Interpretation rule:

- `high`: exact accession lookup from ENA, NCBI SRA/BioSample, or EGA.
- `medium`: curated/prefix-derived label; useful for smoke tests and hypothesis generation.
- `low`: unresolved; do not trust for final correction.

Important warning: a true study label can still be unsafe if it is nearly identical to phenotype. Use `phenotype_confounding_warning` to avoid removing biological disease signal.

## Batch Diagnostics To Add Next

Add a notebook module called `Batch Effect Diagnostics From Learned Embeddings`:

1. Extract sample embeddings from pretrained/gut-adapted checkpoint.
2. Train shallow batch probe:
   - input: sample embeddings
   - target: `batch_label_external_recommended`
   - metric: balanced accuracy, macro-F1
3. Train phenotype probe:
   - input: same embeddings
   - target: phenotype or H/D
   - metric: macro-F1/AUROC where applicable
4. Compute batch-phenotype confounding:
   - contingency table
   - Cramer's V
   - normalized mutual information
   - top phenotype fraction per batch
5. Plot UMAP colored by batch and phenotype.
6. If adding adversarial correction, compare before/after:
   - batch probe should decrease
   - phenotype/H-D probe should remain stable or improve

## Required Data Files

The package preserves the `dataset_v3/` layout. Required files include:

- `abund_pretraining_phase1_gut_and_nongut.csv.zip`
- `abund_pretraining_phase2_gut.csv.zip`
- `abund_finetuning_gut_prev3.csv.zip`
- `meta_pretraining_phase1_gut_and_nongut.csv`
- `meta_pretraining_phase2_gut.csv`
- `meta_finetuning_gut_prev3.csv`
- `species_taxonomy_filled_validated_Serena.xlsx`
- `ExVal/df_validation_data.csv`
- `ExVal/df_validation_data_metadata.csv`

## Professor-Facing Output Standard

Always distinguish:

- smoke-test evidence: code path executes on reduced samples,
- scientific evidence: full intended training/evaluation run,
- unresolved uncertainty: label ambiguity, phenotype-batch confounding, one-class predictions, or suspicious calibration.
