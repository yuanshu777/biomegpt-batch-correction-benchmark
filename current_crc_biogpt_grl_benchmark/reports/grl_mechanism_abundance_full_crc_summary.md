# Mechanism-Only GRL Abundance Decoder Sweep

## Scope

This sweep avoids directly optimizing final MMUPHin benchmark metrics. It changes only the correction mechanism:

- condition label is used as adversary context, not as a strong condition-classifier objective
- stronger reconstruction/preservation
- relative-abundance preservation
- feature-variance preservation to reduce low-dimensional collapse
- optional study-conditioned decoder side channel

All outputs are non-negative 484-species abundance matrices evaluated by the original R benchmark:

- `C:\Users\Yuanshu\Documents\new_attemp_batch\evaluate_crc_method.R`
- Bray-Curtis PERMANOVA
- fixed study prediction folds
- fixed disease LOSO folds
- glmnet classifier

## Results

| method | Study R2 ctrl condition | Study BA | Disease LOSO AUC | Condition R2 ctrl study |
|---|---:|---:|---:|---:|
| raw | 0.078563 | 0.756370 | 0.709835 | 0.007875 |
| mmuphin | 0.030046 | 0.673665 | 0.687841 | 0.008849 |
| grl_abundance_l8_lam10_cw01_rw1 | 0.032837 | 0.553112 | 0.882391 | 0.072273 |
| grl_abundance_l8_lam10_cw001_rw1 | 0.033063 | 0.529881 | 0.667158 | 0.010930 |
| grl_abundance_l8_lam10_cw0_rw1 | 0.052688 | 0.506906 | 0.674264 | 0.018473 |
| grl_mech_context_only_l8_lam10_rw5_rel1_var1 | 0.011405 | 0.377597 | 0.688281 | 0.023348 |
| grl_mech_studydec_context_l8_lam10_rw5_rel1_var1 | 0.318630 | 0.994441 | 0.614904 | 0.009849 |
| grl_mech_context_only_l16_lam10_rw5_rel1_var1 | 0.048439 | 0.487505 | 0.663474 | 0.016299 |

## Interpretation

The best mechanism-only result is:

`grl_mech_context_only_l8_lam10_rw5_rel1_var1`

It improves over MMUPHin on the two study-removal metrics:

- Study R2: `0.011405` vs MMUPHin `0.030046`
- Study BA: `0.377597` vs MMUPHin `0.673665`

It also preserves disease LOSO AUC at essentially the MMUPHin level:

- Disease LOSO AUC: `0.688281` vs MMUPHin `0.687841`

The remaining caveat is condition R2:

- Condition R2: `0.023348` vs MMUPHin `0.008849`

This is much less suspicious than the earlier `cw0.1` result, where condition R2 reached `0.072273`, but it is still elevated. Therefore this is the first genuinely promising GRL abundance-decoder candidate, not yet a final claim.

The study-conditioned decoder setting failed badly in this configuration: it produced high study R2 and near-perfect study classifier balanced accuracy, so this side-channel is not helpful as currently implemented.

## Current Reading

Do not claim final superiority yet, but the mechanism-only context setup is the strongest result so far.

Fair wording:

> A mechanism-only GRL abundance decoder, using condition only as adversary context plus stronger abundance preservation, reduced study signal more than MMUPHin while maintaining disease LOSO AUC at the MMUPHin level. Condition R2 remains elevated and requires follow-up tuning/validation.

## Next Tuning Direction

Stay near the best setting:

- `latent_dim=8`
- `lambda_grl=10`
- `condition_head=False`
- `condition_context_for_adversary=True`
- `recon_weight=5`
- `rel_recon_weight=1`
- `variance_weight=1`

Next try small changes to reduce condition R2 without losing AUC:

- `variance_weight=2`
- `rel_recon_weight=2`
- `lambda_grl=8`
- `hidden_dim=64`
- multiple seeds for the best candidate

## Output Files

- comparison table: `outputs\metrics\grl_mechanism_abundance_full_crc_r_evaluator_comparison.csv`
- training script: `scripts\train_grl_abundance_decoder_full_crc.py`
- best candidate matrix: `C:\Users\Yuanshu\Documents\new_attemp_batch\crc_controlled_benchmark\methods\scgpt_biomegpt\grl_mech_context_only_l8_lam10_rw5_rel1_var1.csv`
- R evaluator report: `C:\Users\Yuanshu\Documents\new_attemp_batch\crc_controlled_benchmark\reports\methods\grl_mech_context_only_l8_lam10_rw5_rel1_var1`
