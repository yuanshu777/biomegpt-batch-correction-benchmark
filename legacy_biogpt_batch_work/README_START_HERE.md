# BiomeGPT VS Code SSH Package

Open this folder in VS Code on the SSH machine. The goal is to continue the BiomeGPT-style microbiome foundation model work without losing project context.

## What To Open First

1. Read `CODEX_PROJECT_CONTEXT.md`.
2. Run `python scripts/bootstrap_smoke_check.py` to verify files.
3. Use `dataset_v3/BiomeGPT_full_pipeline_vscode_ssh.ipynb` for VS Code SSH.
4. Keep `dataset_v3/BiomeGPT_full_pipeline_colab.ipynb` as the original Colab-oriented notebook.

## Compute Rule

Use local/SSH GPU only for smoke tests, reduced debugging, validation, and refactoring. Full pretraining/fine-tuning should be run intentionally later, not accidentally.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_vscode.txt
# install torch according to your server CUDA version, then:
python scripts/bootstrap_smoke_check.py
```

If the notebook cannot find data, set:

```bash
export BIOMEGPT_DATA_DIR=/absolute/path/to/dataset_v3
```
