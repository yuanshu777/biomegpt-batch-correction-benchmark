# Taxonomy-Aware BiomeGPT Reproduction Plan and Report

## Research Goal

This workflow asks what can be learned from the pretrained BiomeGPT species model before and after supervised fine-tuning. The key idea is that the model should learn two interpretable representation spaces: sample embeddings, analogous to cell embeddings in scGPT, and species embeddings, analogous to gene embeddings in scGPT.

## Data Inventory

The phase-1 pretraining matrix contains 1,012 species, and all 1,012 have taxonomy entries in `species_taxonomy_filled_validated_Serena.xlsx`. The `_prev3` fine-tuning matrix contains 513 gut-filtered species. The external validation matrix contains 927 samples and overlaps with 512 of those 513 `_prev3` species, so the evaluation can be performed with only one missing training species set to zero in ExVal.

## Taxonomic Hierarchy Implementation

The original species token is replaced by a sum of rank-specific embeddings:

`Domain + Kingdom + Phylum + Class + Order + Family + Genus + Species`.

This makes taxonomy part of the inductive bias. Species from the same genus share the same genus embedding; genera from the same family share the same family embedding; and so on. Biomedical interpretation becomes easier because clusters in species-embedding space can be read as learned taxonomic organization rather than arbitrary token proximity.

## Species Prompt

The species prompt returns the learned species representation from the taxonomy-composed embedding table. It is analogous to the gene embedding in scGPT. UMAPs colored by genus, family, or order test whether the model organizes microbial taxa in a biologically coherent way. If same-genus or same-family organisms cluster, the model has internalized taxonomic structure that can support biomarker interpretation.

## Sample Prompt

The sample prompt returns the final `<cls>` embedding for a microbiome sample. It is analogous to the cell embedding in scGPT. A gut vs non-gut UMAP tests whether unsupervised pretraining learns body-site structure without explicit labels. Clear separation suggests that BiomeGPT captures community-level ecological signatures, not just individual high-abundance taxa.

## Healthy vs Diseased Fine-Tuning

The downstream task fine-tunes the phase-2 gut-adapted model on `_prev3` gut data and evaluates on the independent ExVal cohort. Healthy samples are labeled H and every non-healthy phenotype is labeled D. The main metric is macro-F1 because the external validation set is imbalanced and macro-F1 gives equal importance to H and D performance.

The pipeline reports accuracy, F1, AUROC, macro-accuracy, macro-F1, macro-AUROC, H accuracy, and D accuracy. H accuracy is the true-negative rate for Healthy samples. D accuracy is the true-positive rate for Diseased samples.

## Class Imbalance and Synthetic Diseased Samples

The training set contains more Healthy than Diseased samples. The pipeline augments the Diseased class by adding Gaussian noise to nonzero abundance entries only, clips negative values to zero, and preserves zero-abundance species as zero. This follows the paper's augmentation principle and avoids inventing species that were absent in a real sample.

## Biomedical Interpretation

Good ExVal macro-F1 would suggest that the pretrained representation transfers across study cohorts and captures disease-associated microbial configurations robust to dataset shift. Per-class accuracy is essential: high overall accuracy can hide poor Diseased recall when Healthy samples dominate. Species UMAPs and attention/embedding analyses can then identify whether performance is driven by broad ecological signals, taxonomic families, or disease-enriched species groups.

The gut vs non-gut sample UMAP is a sanity check for ecological representation learning. If stool samples separate from oral, skin, and vaginal samples without body-site labels during pretraining, the model has learned high-level microbiome niche structure. The taxonomy-colored species UMAP is a sanity check for biological organization in token space. If species from the same genus, family, or order cluster, the hierarchy embeddings are shaping the latent space in a way that should improve interpretability and may improve transfer to species-sparse external cohorts.

## Current Implementation Status

Implemented in `dataset_v3/biomegpt_taxonomy_pipeline.py`:

- taxonomy-aware BiomeGPT model
- rank-wise embedding composition
- species prompt extraction
- sample prompt extraction
- phase-1/phase-2 pretraining command
- UMAP/PCA export for samples and species
- Healthy vs Diseased fine-tuning
- Diseased-class synthetic augmentation with zero-preserving noise
- optional L1/Lasso-style feature selection
- threshold optimization over 0.1 to 0.9
- ExVal metrics and prediction export

## Recommended Commands

Install optional dependencies:

```powershell
pip install -r dataset_v3\requirements_taxonomy_pipeline.txt
```

Pretrain taxonomy-aware model:

```powershell
python dataset_v3\biomegpt_taxonomy_pipeline.py pretrain --data_dir dataset_v3 --taxonomy_xlsx species_taxonomy_filled_validated_Serena.xlsx --output_dir dataset_v3\outputs_taxonomy_biomegpt --epochs_phase1 30 --epochs_phase2 10 --batch_size 64 --mixed_precision
```

Extract sample and species UMAPs:

```powershell
python dataset_v3\biomegpt_taxonomy_pipeline.py embeddings --data_dir dataset_v3 --taxonomy_xlsx species_taxonomy_filled_validated_Serena.xlsx --checkpoint dataset_v3\outputs_taxonomy_biomegpt\taxonomy_checkpoint_stage2.pt --output_dir dataset_v3\outputs_taxonomy_biomegpt --sample_umap --species_umap
```

Fine-tune Healthy vs Diseased and evaluate ExVal:

```powershell
python dataset_v3\biomegpt_taxonomy_pipeline.py finetune_hd --data_dir dataset_v3 --exval_dir ExVal --taxonomy_xlsx species_taxonomy_filled_validated_Serena.xlsx --checkpoint dataset_v3\outputs_taxonomy_biomegpt\taxonomy_checkpoint_stage2.pt --output_dir dataset_v3\outputs_taxonomy_biomegpt --epochs 20 --batch_size 64 --augment_diseased --synthetic_std 5 --use_l1_feature_selection
```
