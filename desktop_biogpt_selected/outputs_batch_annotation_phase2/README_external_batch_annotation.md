# External Batch Annotation Sources

This folder enriches `meta_pretraining_phase2_gut.csv` using exact public accession lookups where possible.

Primary external sources used:
- ENA Portal API for `SAMN`, `SAMEA`, and `SAMD` sample-to-study metadata.
- NCBI SRA RunInfo for `SRR` run-to-BioProject/BioSample metadata.
- NCBI BioSample E-utilities fallback for old `SAMN` accessions where ENA lacked `study_accession`.
- EGA Metadata API for `EGAR` run-to-EGA-study metadata.

Curated prefix-derived labels are explicitly not exact accession lookups. They remain `medium` confidence and `needs_manual_review=True`.

Conservative rule for final batch correction:
- exact accession lookup required, and
- batch group must not be strongly phenotype-confounded (`top_phenotype_fraction < 0.80` and more than one phenotype).
