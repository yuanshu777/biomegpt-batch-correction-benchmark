# BiomeGPT Core Notes

This folder is a minimal scaffold for the final CRC overlap benchmark. It does
not restore all historical BiomeGPT experiments.

Current supported local path:

1. If an existing raw CLS embedding matrix is supplied in `configs/paths.yaml`,
   subset it to the 389 overlap samples.
2. If only a checkpoint is supplied, the script fails with a clear message until
   the current BiomeGPT architecture/checkpoint contract is provided.

Expected future CLS output format:

- sample x dimension CSV
- first column: `sample_id`
- rows must use MMUPHin CRC sample IDs after mapping from BiomeGPT sample IDs

