# Pipeline Status

Run mode: **SMOKE TEST**

Implemented stages:
- Data alignment and contracts
- Taxonomy cleaning and rank-wise encoding
- Species prompt extraction
- Sample prompt extraction
- Representation analysis with plots and quantitative summaries
- Healthy vs Diseased fine-tuning from the phase-2 checkpoint
- Training-side threshold optimization for macro-F1
- ExVal evaluation and report artifacts

Data contract snapshot:
- Phase1 species: 1012
- Phase2 species: 875
- `_prev3` species: 513
- ExVal species: 2602
- `_prev3` species missing from ExVal: ['Leuconostoc_virus_P793']

Interpretation note: smoke-test outputs verify code paths and file contracts only. They are not scientific performance estimates.
