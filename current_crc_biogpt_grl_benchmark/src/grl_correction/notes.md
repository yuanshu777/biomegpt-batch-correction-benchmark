# GRL / scGPT-style Correction Notes

This module is an embedding-level correction scaffold for the CRC/MMUPHin
benchmark. It is not full BiomeGPT pretraining and it does not claim that the
method works yet.

## Input Contract

Minimum input:

- sample x embedding CSV with `sample_id`
- metadata with `sample_id`, `studyID`, and `study_condition`

The intended first real input is
`outputs/crc_overlap_benchmark/biogpt_raw_cls_389.csv`, extracted from the
BiomeGPT taxonomy checkpoint for the 389 MMUPHin-overlap samples.

## Version 1: Preservation + External Probes

The current trainer adds:

- gradient reversal study adversary
- CRC/control condition classifier
- class-weighted cross entropy for study and condition labels
- mini-batch training through a PyTorch `DataLoader`
- GRL lambda schedules: constant, linear warmup, and DANN-style warmup
- preservation loss so corrected embeddings cannot drift arbitrarily
- external sklearn probes for study and condition signal when sklearn is
  available

If `latent_dim == input_dim`, preservation is `MSE(z, standardized_x)`. If
dimensions differ, a small decoder reconstructs the standardized input.

## Version 2: Study-conditioned Decoder

The trainer can optionally use `StudyConditionedDecoder`, where decoder input is
`z + study_embedding` and the reconstruction target is the original standardized
embedding. This gives study information a side channel for reconstruction while
the corrected representation `z` is discouraged from encoding study.

The optional condition-aware adversary concatenates a condition embedding to the
study adversary input. GRL is applied only to `z`, not to the condition
embedding. The scientific intent is to remove residual study signal while
protecting CRC/control condition structure.

## Version 2b: NoMean CLS Adapter Prototype

`nomean_adapter.py` adds a more conservative CLS-specific prototype. It starts
from raw BiomeGPT CLS embeddings and learns a small residual correction:

```text
z = h + 0.1 * Adapter(h)
```

This version keeps the full CLS dimension, does not do study mean-centering, and
does not use a condition classifier. It uses:

- reconstruction of raw CLS through a study-aware decoder side channel
- direct preservation loss between `z` and standardized raw CLS
- weak conditional study GRL, where condition is context for the adversary
- variance and covariance preservation penalties to reduce collapse risk

The first use is `scripts/run_nomean_cls_adapter_crc389.py`, which evaluates the
389-sample overlap benchmark against raw abundance, MMUPHin adjusted abundance,
raw BiomeGPT CLS, and study-mean-centered CLS. This is still a local
full-data/transductive prototype; promising settings require LOSO or
cross-fitted correction before any claim.

`scripts/run_nomean_grl_ablation_crc389.py` compares five GRL formulations on
the same adapter:

- vanilla study-ID GRL
- conditional GRL with condition context
- residual conditional GRL with a condition-only study prior
- pairwise within-condition same-study GRL
- residual conditional GRL plus conditional CORAL alignment

These are mechanism ablations, not final claims. The purpose is to see whether
changing the adversarial question can reduce study leakage without low-rank
collapse or condition-label amplification.

## Future Version 3

Potential future extensions:

- zero-inflation-aware abundance reconstruction
- decoder targets that combine embeddings and abundance vectors
- study embedding shrinkage or regularization
- stronger train/validation split handling for external probes

These should be added only after the current local scaffold is evaluated on the
389-sample CRC overlap benchmark.
