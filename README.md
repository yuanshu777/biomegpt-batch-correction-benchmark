# GitHub-Ready Code Package: BiomeGPT / scGPT-Style Batch Correction

Prepared: 2026-07-06

This folder is a sanitized, code-focused package assembled from three working areas:

- `C:\Users\Yuanshu\Documents\new_attemp_batch`
- `C:\Users\Yuanshu\Documents\ali lab\scgpt`
- `C:\Users\Yuanshu\Desktop\Ali lab`

It is intended for GitHub upload or handoff to another developer. It deliberately excludes large datasets, checkpoints, virtual environments, cached files, and very large historical zip archives.

## Contents

### `current_crc_biogpt_grl_benchmark/`

Current CRC/MMUPHin benchmark package. This is the most important active codebase.

Expected contents:

- `src/`: BiomeGPT checkpoint loading, CLS extraction, GRL correction modules, evaluation helpers, MMUPHin bridge utilities.
- `scripts/`: reproducible scripts for MMUPHin CRC benchmark setup, BiomeGPT CLS extraction, GRL experiments, residual GRL, cross-fitting, and diagnostics.
- `configs/`: local path and benchmark config files.
- `reports/`: markdown/csv summaries of the main experiments.
- `data_manifest/`: overlap manifest for CRC389.
- `README.md`: package-level notes.

### `scgpt_local_changes/`

Local scGPT-related changes and minimal pipeline code.

This is not a full vendored copy of upstream scGPT. The original repo remote was:

```text
https://github.com/bowang-lab/scGPT.git
```

Included here:

- `scgpt_local_changes.patch`: local diff against upstream checkout.
- `patched_files/`: copies of locally modified tracked files.
- `pipelines/minimal_atlas/`: local minimal atlas pipeline.
- `examples/`: local smoke/example scripts.
- `scgpt_pretrain_colab_clean.ipynb`: cleaned Colab notebook.

### `legacy_biogpt_batch_work/`

Earlier BiomeGPT-style batch-correction work from the Desktop Ali lab folder. This is useful historical context, not the current canonical MMUPHin CRC benchmark.

### `desktop_biogpt_selected/`

Selected BiomeGPT notebooks, scripts, reports, and lightweight summaries from the older Desktop Ali lab working folder.

### `shared_memory/`

Small project-map/context files that explain how the scGPT and BiomeGPT folders were related.

## Main Scientific Status

The current reliable result is not "GRL beats MMUPHin."

Safer summary:

> The work establishes a controlled MMUPHin CRC benchmark and shows that BiomeGPT CLS embeddings contain measurable study/cohort signal. Simple post-hoc GRL reduces study classifier predictability but does not reliably remove global study-associated variance. The stronger next direction is batch-aware BiomeGPT adapter/LoRA post-training with invariant/nuisance splitting, study-conditioned reconstruction, conditional CORAL/MMD, and strict LOSO/cross-fitted evaluation.

## What Was Excluded

Excluded from this GitHub-ready package:

- raw data directories;
- `.pt` checkpoints;
- `.venv` environments;
- `.git` folders;
- `__pycache__` and test caches;
- large zip archives;
- large generated matrix outputs;
- most PDF/image-heavy report artifacts.

If rerunning BiomeGPT CLS extraction is required, the checkpoint files should be shared separately.

## Suggested GitHub Setup

After reviewing this folder:

```powershell
cd C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\github_ready_code_20260706
git init
git add .
git commit -m "Add BiomeGPT batch-correction handoff code"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Do not push directly to the upstream scGPT repo. Create a new private repo or a lab-owned repo first.

## Recommended Next Development Direction

1. Use the full551 MMUPHin CRC benchmark as the canonical benchmark.
2. Treat CRC389 as an exploratory overlap subset only.
3. Implement true model-level batch-aware adaptation:
   - BiomeGPT backbone plus adapter/LoRA;
   - invariant CLS representation;
   - nuisance/study side-channel;
   - masked abundance reconstruction;
   - residual conditional adversary;
   - conditional CORAL/MMD;
   - anti-collapse geometry preservation.
4. Evaluate with LOSO or cross-fitted correction.
5. Add additional multi-study disease benchmarks before making broad claims.

