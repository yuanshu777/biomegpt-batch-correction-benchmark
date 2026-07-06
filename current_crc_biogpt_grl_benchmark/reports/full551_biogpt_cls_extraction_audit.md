# Full551 BiomeGPT CLS Extraction Audit

This checks whether the original 551 MMUPHin CRC abundance table can be directly mapped into the stage2 BiomeGPT checkpoint species vocabulary.

## Mapping Summary

```json
{
  "checkpoint_error": null,
  "mmuphin_species_count": 484,
  "checkpoint_species_count": 1012,
  "mapped_species_count": 357,
  "mapped_species_fraction": 0.737603305785124,
  "unmapped_species_count": 127,
  "per_sample_nonzero_mapped_min": 1,
  "per_sample_nonzero_mapped_median": 98.0,
  "per_sample_nonzero_mapped_max": 172,
  "cls_extraction_feasible": true,
  "cls_result": {
    "status": "extracted",
    "output": "C:\\Users\\Yuanshu\\Documents\\new_attemp_batch\\crc_biogpt_grl_benchmark\\outputs\\crc_full551_benchmark\\biogpt_raw_cls_551.csv",
    "n_samples": 551,
    "n_dimensions": 512,
    "mapped_checkpoint_species": 357
  }
}
```

## Reading

- If `biogpt_raw_cls_551.csv` exists, CLS extraction succeeded on the canonical 551 benchmark.
- If extraction failed, the JSON reason gives the exact blocker.
- The mapping audit CSV lists every MMUPHin species and whether it maps to a checkpoint species after conservative normalization.
