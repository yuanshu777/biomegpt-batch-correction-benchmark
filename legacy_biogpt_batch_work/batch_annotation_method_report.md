# Batch Annotation Method Report

Date: 2026-05-20

Input metadata: `meta_pretraining_phase2_gut.csv`

Output annotation: `meta_pretraining_phase2_gut_batch_annotation_external_enriched.csv`

## Executive Summary

The original phase-2 gut metadata file did not contain an explicit sequencing batch, study ID, cohort ID, sequencing center, platform, or country column. It contained only:

```text
sample_id
Phenotype
body_site
Phenotype_fullname
```

Therefore, batch labels were reconstructed from sample identifiers using a conservative, provenance-tracked strategy:

1. Public accessions were mapped to external study/cohort metadata using public databases.
2. Local non-accession IDs were labeled only by prefix-derived cohort rules.
3. Every label was assigned a confidence level.
4. Every batch group was checked for phenotype confounding.
5. Only exact external accession mappings that were not strongly phenotype-confounded were marked as conservative-safe for final batch correction.

The final enriched annotation contains 13,332 samples and 218 recommended batch labels.

## Final Counts

### Sample ID Types

| ID type | Samples | Interpretation |
|---|---:|---|
| `BioSample_SAMEA` | 3,667 | ENA/EBI BioSample accession |
| `BioSample_SAMN` | 3,573 | NCBI BioSample accession |
| `EGA_EGAR` | 1,490 | EGA run accession |
| `SRA_SRR` | 442 | NCBI SRA run accession |
| `BioSample_SAMD` | 217 | DDBJ/INSDC BioSample accession |
| `local_or_unknown` | 3,943 | Local sample naming pattern, no direct public accession |

### External Confidence

| Confidence | Samples | Meaning |
|---|---:|---|
| `high` | 9,389 | Exact public accession lookup returned study/cohort metadata |
| `medium` | 3,418 | Curated or prefix-derived cohort label; useful but not exact public accession proof |
| `low` | 525 | Unresolved local ID pattern |

### Data Source Used

| Source | Samples |
|---|---:|
| ENA Portal API sample+study lookup | 7,388 |
| EGA Metadata API run+study lookup | 1,490 |
| NCBI SRA RunInfo lookup | 442 |
| NCBI BioSample E-utilities fallback | 69 |
| Curated prefix-derived local cohort label | 2,147 |
| Prefix-derived local cohort label with external/name knowledge | 1,271 |
| Unresolved local sample ID pattern | 525 |

### Conservative-Safe Labels

| Flag | Samples | Meaning |
|---|---:|---|
| `safe_for_final_batch_correction_conservative=True` | 2,134 | Exact external accession mapping and not strongly phenotype-confounded |
| `safe_for_final_batch_correction_conservative=False` | 11,198 | Either uncertain/prefix-derived or strongly phenotype-confounded |

This strict filtering is intentional. A true study label is not automatically safe for batch correction if the study is nearly identical to a disease phenotype.

## Why Batch Annotation Had To Be Reconstructed

The file `meta_pretraining_phase2_gut.csv` is a gut-only metadata table with 13,332 stool samples. It includes disease/phenotype labels but not technical or study-level batch labels. Because scGPT-style batch correction requires a batch/domain label, we needed to infer the best available batch proxy from sample IDs.

The sample IDs contain a mixture of:

```text
SAMEA104335965
SAMN03283239
SAMD00114969
SRR16124168
EGAR00001420100_9002000001328080LL
M01.1-V1-stool
PRIMM0541
wHAXPI032581-18
SID5420-2
```

The public accession-like IDs can be queried externally. The local IDs cannot be validated through public accession APIs directly, so they were treated more cautiously.

## External Sources

The external lookup relied on official public metadata services:

| Source | Used for | Official reference |
|---|---|---|
| ENA Portal API | `SAMN`, `SAMEA`, `SAMD` sample-to-study metadata | [ENA programmatic access](https://ena-docs.readthedocs.io/en/latest/retrieval/programmatic-access.html), [ENA advanced search / Portal API](https://ena-browser-docs.readthedocs.io/en/latest/browser/search/advanced.html) |
| NCBI SRA RunInfo | `SRR` run-to-BioProject/BioSample metadata | [NCBI SRA](https://www.ncbi.nlm.nih.gov/sra/) |
| NCBI E-utilities / BioSample | fallback for old `SAMN` BioSample records missing ENA study accessions | [NCBI E-utilities Help](https://www.ncbi.nlm.nih.gov/sites/books/NBK25501/) |
| EGA Metadata API | `EGAR` run-to-study metadata | [EGA Metadata Public API](https://ega-archive.org/discovery/metadata/public-metadata-api/) |

## Step 1: Parse Sample ID Type

Each `sample_id` was classified by pattern matching:

```text
^SAMN\d+   -> BioSample_SAMN
^SAMEA\d+  -> BioSample_SAMEA
^SAMD\d+   -> BioSample_SAMD
^SRR\d+    -> SRA_SRR
^EGAR\d+   -> EGA_EGAR
otherwise -> local_or_unknown
```

For IDs such as:

```text
EGAR00001420100_9002000001328080LL
```

only the public run accession portion was extracted:

```text
EGAR00001420100
```

The full original sample ID was preserved.

## Step 2: ENA Lookup For `SAMN`, `SAMEA`, and `SAMD`

For BioSample-style accessions, ENA Portal API was queried using exact `sample_accession` matching.

The relevant ENA fields were:

```text
sample_accession
secondary_sample_accession
study_accession
sample_title
description
first_public
last_updated
center_name
country
host_body_site
host_status
disease
host
host_scientific_name
project_name
collection_date
```

The study accessions were then queried against ENA study metadata to retrieve:

```text
study_accession
secondary_study_accession
study_title
study_description
center_name
study_name
project_name
```

Example mapping:

```text
SAMEA104335965
-> PRJEB103248;PRJEB22893
-> Understanding the gut microbiome in melanoma patients
```

This category produced 7,388 high-confidence labels through ENA.

## Step 3: NCBI SRA RunInfo Lookup For `SRR`

For `SRR` run IDs, the NCBI SRA RunInfo endpoint was used. The fields used included:

```text
Run
Experiment
LibraryName
LibraryStrategy
Platform
Model
SRAStudy
BioProject
Sample
BioSample
SampleName
CenterName
Submission
Consent
```

The preferred batch label was:

```text
study:<BioProject>
```

If BioProject was unavailable, the fallback was SRAStudy.

Example:

```text
SRR16124168
-> BioProject PRJNA763023
-> batch_label_external_recommended = study:PRJNA763023
```

This category produced 442 high-confidence labels.

## Step 4: NCBI BioSample Fallback For Older `SAMN` Records

Some older `SAMN` BioSample records were visible in ENA but did not have a usable ENA `study_accession`. These were not discarded. Instead, NCBI E-utilities were used:

```text
ESearch db=biosample term=<SAMN>[Accession]
EFetch  db=biosample id=<uid> retmode=xml
```

The BioSample XML was parsed for fields such as:

```text
gap_accession
study_name
study_design
submitter_handle
SRA accession
sample title
```

Example:

```text
SAMN00143400
-> gap_accession = phs000228
-> study_name = HMP Core Microbiome Sampling Protocol A (HMP-A)
-> batch_label_external_recommended = dbgap:phs000228
```

This fallback resolved 69 additional samples to high confidence.

## Step 5: EGA Metadata Lookup For `EGAR`

For EGA run accessions, the EGA Metadata API was queried by run ID.

The lookup retrieved:

```text
run accession
EGA sample accession
EGA study accession
EGA study title
EGA study description
released date
```

Example:

```text
EGAR00001420100
-> EGAS00001001704
-> LifeLines-DEEP population multi-omix cohort
-> batch_label_external_recommended = ega_study:EGAS00001001704
```

This category produced 1,490 high-confidence labels.

## Step 6: Prefix-Derived Local Cohort Labels

For local IDs without public accessions, exact external database confirmation was not possible from this metadata alone. These were assigned prefix-derived labels only when the naming pattern was structured and biologically interpretable.

Examples:

| Prefix / pattern | Assigned label | Confidence | Reason |
|---|---|---|---|
| `M...` | `prefix:MetaCardis cardiometabolic disease cohort` | medium | Large cardiometabolic phenotype mixture: T2D, IGT, CAD, HF, MS, Healthy |
| `PRIMM...` | `prefix:PRIMM melanoma immunotherapy cohort` | medium | PRIMM naming pattern and melanoma phenotype |
| `SID...` | `prefix:SID_healthy_local_cohort` | medium | Structured local healthy cohort prefix |
| `wHAXPI...` | `prefix:wHAXPI_schizophrenia_local_cohort` | medium | Structured prefix with Schizo/Healthy labels |
| `CM...` | `prefix:CM_STH_local_cohort` | medium | Structured prefix with STH/Healthy labels |
| `FMT...` | `prefix:FMT_CDI_local_cohort` | medium | Structured FMT/CDI prefix |
| `Travel...` | `prefix:international_travel_stool_cohort` | medium | Structured travel cohort prefix |

These labels are useful for diagnostics and smoke testing, but they are not treated as final verified batch labels.

A total of 3,418 samples received medium-confidence prefix-derived labels.

## Step 7: Unresolved Local IDs

The remaining 525 samples had local sample IDs that were not confidently mapped to a study/cohort.

These rows were marked:

```text
external_confidence = low
external_source = unresolved local sample ID pattern
needs_manual_review = True
```

They were also exported to:

```text
meta_pretraining_phase2_gut_unresolved_batch_review_queue.csv
```

Examples include UUID-like local IDs and small local prefixes that were not sufficiently interpretable from `meta_pretraining_phase2_gut.csv` alone.

## Step 8: Recommended Batch Label Construction

The main usable field is:

```text
batch_label_external_recommended
```

It was constructed as follows:

| Source type | Label format |
|---|---|
| ENA sample/study lookup | `study:<study_accession>` |
| NCBI SRA RunInfo | `study:<BioProject>` |
| NCBI BioSample/dbGaP fallback | `dbgap:<phs_accession>` |
| EGA Metadata API | `ega_study:<EGAS_accession>` |
| Curated local prefix | `prefix:<curated_cohort_name>` |
| Unresolved local prefix | `needs_review:<prefix>` |

When a sample mapped to multiple study accessions, accessions were joined with `+`, for example:

```text
study:PRJEB103728+PRJEB11532
```

This happens frequently when ENA includes both the original project and a downstream TPA/metagenomic assembly project.

## Step 9: Confidence Rules

### High Confidence

A row was marked high confidence if:

```text
The sample_id contained a public accession,
and an exact external database lookup returned study/cohort metadata.
```

Examples:

```text
SAMEA -> ENA study_accession
SAMN  -> ENA study_accession or NCBI BioSample/dbGaP fallback
SAMD  -> ENA study_accession
SRR   -> NCBI BioProject/SRAStudy
EGAR  -> EGA study accession
```

### Medium Confidence

A row was marked medium confidence if:

```text
No exact public accession lookup was possible,
but the local sample ID prefix was structured and matched a coherent cohort/study pattern.
```

Medium-confidence labels should be used for:

```text
diagnostics
smoke tests
hypothesis generation
manual review queues
```

They should not be treated as final verified batch labels.

### Low Confidence

A row was marked low confidence if:

```text
No public accession was available,
and the local ID pattern was too weak or ambiguous for a defensible cohort assignment.
```

These rows require manual review before any serious batch-correction use.

## Step 10: Phenotype-Confounding Check

Batch correction is scientifically dangerous if the proposed batch label is almost the same thing as the disease label.

For each proposed batch label, we computed:

```text
n_samples
n_phenotypes
top_phenotype
top_phenotype_count
top_phenotype_fraction
```

The warning rule was:

```text
phenotype_confounding_warning = True
if top_phenotype_fraction >= 0.80
or n_phenotypes <= 1
```

This means that if a batch group is almost entirely one phenotype, then removing batch signal may also remove biological disease signal.

Example high-confidence but unsafe groups:

```text
ega_study:EGAS00001001704
n_samples = 1135
top_phenotype = Healthy
top_phenotype_fraction = 1.0

study:PRJEB61255+PRJNA834801
n_samples = 355
top_phenotype = PD
top_phenotype_fraction = 1.0
```

These are real study labels, but they are not safe for naive adversarial correction because study and phenotype are perfectly or nearly perfectly confounded.

## Step 11: Conservative-Safe Rule

The strict final flag is:

```text
safe_for_final_batch_correction_conservative
```

It is `True` only if:

```text
external_confidence == "high"
and phenotype_confounding_warning == False
```

This produced:

```text
2,134 conservative-safe samples
11,198 not conservative-safe samples
```

This is why the downstream batch-correction experiments should prefer conservative-safe labels. Broad labels may be useful for diagnostics, but they are too phenotype-confounded for direct correction.

## Main Output Columns

The enriched file contains 92 columns. The most important columns are:

| Column | Meaning |
|---|---|
| `sample_id` | Original sample ID from `meta_pretraining_phase2_gut.csv` |
| `Phenotype` | Original disease/healthy phenotype |
| `Phenotype_fullname` | Full phenotype name |
| `accession_type` | Parsed ID type, e.g. `BioSample_SAMN`, `EGA_EGAR`, `local_or_unknown` |
| `external_accession` | Extracted public accession if present |
| `batch_label_external_recommended` | Recommended batch/cohort label |
| `external_confidence` | `high`, `medium`, or `low` |
| `external_source` | Which method/database produced the label |
| `external_study_accession` | Study/project accession if available |
| `external_study_title` | Study/cohort title when available |
| `external_sample_or_biosample` | Linked sample/BioSample/SRA sample accession |
| `external_country` | Country when available from ENA |
| `external_center` | Submitting center when available |
| `batch_n_phenotypes` | Number of phenotypes in that batch group |
| `batch_top_phenotype_fraction` | Fraction of the largest phenotype within that batch |
| `phenotype_confounding_warning` | Whether batch is strongly phenotype-confounded |
| `needs_manual_review` | Whether the row needs manual checking |
| `safe_for_final_batch_correction_conservative` | Strict safe flag for final correction experiments |

## Files Produced

### Primary Annotation

```text
meta_pretraining_phase2_gut_batch_annotation_external_enriched.csv
```

This is the main file to use in modeling.

### Batch-Level Summary

```text
meta_pretraining_phase2_gut_batch_external_summary.csv
```

This summarizes each batch label, including phenotype composition and confounding risk.

### Unresolved Review Queue

```text
meta_pretraining_phase2_gut_unresolved_batch_review_queue.csv
```

This lists low-confidence local IDs that still need manual study/cohort resolution.

### Phenotype-Confounding Review Queue

```text
meta_pretraining_phase2_gut_batch_phenotype_confounding_review_queue.csv
```

This lists batch labels where phenotype composition suggests strong confounding.

### External Lookup Cache

```text
outputs_batch_annotation_phase2/external_lookup_ena_samples.tsv
outputs_batch_annotation_phase2/external_lookup_ena_studies.tsv
outputs_batch_annotation_phase2/external_lookup_ncbi_sra_runinfo.tsv
outputs_batch_annotation_phase2/external_lookup_ncbi_biosample_fallback.tsv
outputs_batch_annotation_phase2/external_lookup_ega_runs_studies.tsv
```

These files preserve the raw external lookup evidence, so the annotation is auditable.

## How To Use This In Modeling

### For Diagnostics

Use:

```text
batch_label_external_recommended
```

This can be used to test whether model embeddings contain batch-associated signal.

Recommended diagnostic subsets:

```text
all high+medium labels
high-only labels
conservative-safe labels
```

### For Adversarial Batch Correction

Use only:

```text
safe_for_final_batch_correction_conservative == True
```

or manually reviewed labels.

This avoids training the model to remove disease signal that is confounded with study identity.

### Do Not Use Blindly

Do not directly use all `batch_label_external_recommended` values for final correction. Many are real study labels but biologically unsafe because phenotype and study are confounded.

## Scientific Interpretation

This annotation should be understood as a batch-proxy reconstruction, not a perfect sequencing batch table.

The high-confidence labels are strong proxies for study/cohort/domain. They are appropriate for asking:

```text
Does the pretrained sample embedding contain study/cohort-associated signal?
```

The conservative-safe subset is appropriate for asking:

```text
Can adversarial correction reduce batch-associated signal without destroying phenotype signal?
```

The annotation is not sufficient to claim:

```text
All technical sequencing batches are perfectly known.
```

That would require original study metadata such as sequencing run date, extraction kit, library prep protocol, sequencing center, sequencing instrument, and processing pipeline.

## Limitations

1. The original metadata file did not include explicit technical batch fields.
2. Study/cohort labels are proxies for batch, not guaranteed technical batch labels.
3. Some public studies are completely phenotype-confounded.
4. Local IDs without public accessions cannot be fully validated from this dataset alone.
5. Prefix-derived labels are useful but should remain manual-review labels.
6. Multiple study accessions can occur when ENA links both original projects and derived TPA/metagenomic assembly projects.

## Bottom-Line Recommendation

For professor-facing results, describe the annotation as:

```text
an externally enriched study/cohort batch-proxy annotation built from public accession metadata and conservative prefix-based local cohort rules.
```

For final batch correction, use:

```text
safe_for_final_batch_correction_conservative == True
```

For diagnostics and exploratory analysis, compare:

```text
high+medium labels
high-only labels
conservative-safe labels
```

This lets us show that the model learned batch-associated structure while avoiding the overclaim that every sample has a perfectly known technical batch.

