# scGPT to BiomeGPT Bridge

## Concept Mapping

| scGPT term | BiomeGPT-style microbiome term |
| --- | --- |
| gene token | species token |
| gene embedding | species prompt |
| cell embedding | sample prompt |
| gene vocabulary | species vocabulary |
| expression value | abundance bin |
| cell-level CLS embedding | sample-level CLS embedding |

## Current Integration Strategy

Keep `scGPT-main` as a reference implementation. Do not merge its package files directly into `dataset_v3`.

Use scGPT design ideas for:

- prompt extraction
- embedding visualization
- model interpretability framing
- professor-facing terminology

Use BiomeGPT/dataset_v3 files for:

- species abundance data
- taxonomy-aware species embeddings
- gut/non-gut representation analysis
- Healthy vs Diseased fine-tuning
- ExVal evaluation

## Why Not Physically Mix Code

The scGPT project has its own package structure, lock file, and virtual environment. Copying or moving its internals into `dataset_v3` would make imports and dependency management harder. The cleaner merge is a shared-memory/documentation layer plus explicit path references.

## Active Notebook

Use:

```text
C:\Users\Yuanshu\Desktop\Ali lab\biogpt\dataset_v3\BiomeGPT_full_pipeline_colab.ipynb
```

This notebook is the active research-facing BiomeGPT pipeline.

