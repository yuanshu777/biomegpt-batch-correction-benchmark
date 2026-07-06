# GRL Abundance Decoder on Original MMUPHin CRC Benchmark

## Scope

This run uses the original full MMUPHin CRC benchmark data, not the CRC389 overlap subset.

The GRL model was changed from an embedding-only output to an abundance-compatible autoencoder:

- input: full CRC raw abundance, 484 species x 551 samples
- training space: `log1p(1000 * abundance)` standardized by species
- encoder: abundance -> bottleneck `z`
- adversary: study prediction through GRL, condition-aware using condition embedding
- decoder: `z -> reconstructed abundance`
- output: non-negative reconstructed 484-species abundance matrix in the exact original feature and sample order

The reconstructed abundance matrices were evaluated by the original R benchmark:

- `C:\Users\Yuanshu\Documents\new_attemp_batch\evaluate_crc_method.R`
- Bray-Curtis PERMANOVA
- frozen study-prediction splits
- frozen disease LOSO splits
- glmnet evaluator

## Primary Results

| method | Study R2 ctrl condition | Study BA | Disease LOSO AUC | Condition R2 ctrl study |
|---|---:|---:|---:|---:|
| raw | 0.078563 | 0.756370 | 0.709835 | 0.007875 |
| mmuphin | 0.030046 | 0.673665 | 0.687841 | 0.008849 |
| grl_abundance_l8_lam10_cw01_rw1 | 0.032837 | 0.553112 | 0.882391 | 0.072273 |
| grl_abundance_l16_lam10_cw01_rw1 | 0.045283 | 0.685531 | 0.935005 | 0.051094 |
| grl_abundance_l8_lam5_cw05_rw1 | 0.043732 | 0.486625 | 0.999715 | 0.050493 |
| grl_abundance_l8_lam10_cw001_rw1 | 0.033063 | 0.529881 | 0.667158 | 0.010930 |
| grl_abundance_l8_lam10_cw0_rw1 | 0.052688 | 0.506906 | 0.674264 | 0.018473 |

## Interpretation

The abundance-compatible path works: the GRL outputs pass the original MMUPHin R evaluator, including feature-order validation and Bray-Curtis PERMANOVA.

The strongest-looking configuration by study classifier is `grl_abundance_l8_lam10_cw01_rw1`: study BA drops from raw 0.756 and MMUPHin 0.674 to 0.553, while study R2 is close to MMUPHin at 0.0328 vs 0.0300. However, condition R2 increases strongly to 0.0723 and disease LOSO AUC jumps to 0.882. That is promising but also suspicious: the condition head may be amplifying disease-label information in the reconstructed abundance.

The more conservative configuration is `grl_abundance_l8_lam10_cw001_rw1`: study BA improves to 0.530 and study R2 remains close to MMUPHin at 0.0331, while condition R2 is much closer to raw/MMUPHin at 0.0109. Its disease LOSO AUC is 0.667, slightly below MMUPHin 0.688 and raw 0.710.

## Current Reading

Do not claim this already beats MMUPHin.

The fair statement is:

> An abundance-decoder GRL can now be evaluated under the original MMUPHin benchmark. Some configurations reduce study predictability more than MMUPHin with similar study R2, but the disease-preservation behavior is not yet clean: stronger condition supervision appears to amplify disease signal, while weaker condition supervision slightly underperforms MMUPHin disease LOSO AUC.

## Recommended Next Step

Tune the abundance decoder objective around the conservative region:

- `latent_dim=8`
- `lambda_grl=10`
- `condition_weight` between `0.01` and `0.1`
- `recon_weight >= 1`

The target should be:

- study R2 close to or below MMUPHin `0.030`
- study BA below MMUPHin `0.674`
- disease LOSO AUC close to raw/MMUPHin, without condition R2 exploding

## Output Files

- training script: `scripts/train_grl_abundance_decoder_full_crc.py`
- reconstructed matrices: `C:\Users\Yuanshu\Documents\new_attemp_batch\crc_controlled_benchmark\methods\scgpt_biomegpt\grl_abundance_*.csv`
- R evaluator reports: `C:\Users\Yuanshu\Documents\new_attemp_batch\crc_controlled_benchmark\reports\methods\grl_abundance_*`
- comparison table: `outputs\metrics\grl_abundance_decoder_full_crc_r_evaluator_comparison.csv`
- training summary: `outputs\metrics\grl_abundance_decoder_full_crc_training_summary.json`
- training history: `outputs\metrics\grl_abundance_decoder_full_crc_training_history.csv`
