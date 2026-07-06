# Upload Manifest

Upload the whole package folder or the `.zip` created next to it. The `dataset_v3/` folder inside the package is intentionally self-contained.

## Must Keep

- `dataset_v3/BiomeGPT_full_pipeline_vscode_ssh.ipynb`
- all CSV/ZIP/XLSX files in `dataset_v3/`
- `dataset_v3/ExVal/`
- `CODEX_PROJECT_CONTEXT.md`
- `README_START_HERE.md`
- `requirements_vscode.txt`
- `scripts/bootstrap_smoke_check.py`

## Optional But Useful

- `dataset_v3/BiomeGPT_full_pipeline_colab.ipynb`
- `dataset_v3/biomegpt_taxonomy_pipeline.py`
- `dataset_v3/outputs_batch_annotation_phase2/`
- `docs/`

## Not Included On Purpose

Large old smoke-output directories are not included because they are not needed to continue work and could confuse new runs.
