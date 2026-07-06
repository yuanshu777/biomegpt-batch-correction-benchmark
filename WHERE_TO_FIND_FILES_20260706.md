# Where To Find Everything

Date: 2026-07-06

This document explains where the BiomeGPT / scGPT / MMUPHin CRC batch-correction materials are located, what is included in the GitHub repository, and what is included in the Teams handoff zip.

## 1. Main Sharing Locations

### GitHub Repository

GitHub URL:

```text
https://github.com/yuanshu777/biomegpt-batch-correction-benchmark
```

Purpose:

- code-first sharing;
- easy online browsing;
- useful for continued development;
- excludes large datasets, checkpoints, virtual environments, cache folders, and very large historical zip files.

Local GitHub-ready folder:

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\github_ready_code_20260706
```

Local GitHub-ready zip:

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\github_ready_code_20260706.zip
```

Current GitHub commit after adding this guide:

```text
c210e5c Add file location and sharing guide
```

### Teams Handoff Package

Recommended zip to upload to Microsoft Teams:

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\teams_handoff_ali_lab_biogpt_scgpt_20260706.zip
```

Purpose:

- direct lab/professor handoff;
- includes the GitHub-ready code zip;
- includes the current CRC/MMUPHin benchmark handoff zip;
- includes professor-facing PDFs;
- includes selected older reports and manageable legacy packages;
- excludes 2GB/3GB historical archives and PyTorch checkpoints.

Local staging folder for the Teams package:

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\teams_handoff_20260706
```

## 2. What Is In The GitHub Repository

The GitHub repository is a sanitized, code-focused package. It contains the current benchmark code plus selected historical code needed to understand the development path.

### `current_crc_biogpt_grl_benchmark/`

This is the most important current codebase.

It contains:

- MMUPHin CRC benchmark loading and setup;
- BiomeGPT checkpoint loading and CLS extraction code;
- GRL correction modules;
- NoMean adapter code;
- split / invariant-nuisance adapter code;
- evaluation metrics;
- CRC389 and full551 scripts;
- full551 benchmark reports;
- smoke tests.

Important subfolders:

```text
current_crc_biogpt_grl_benchmark/src
current_crc_biogpt_grl_benchmark/scripts
current_crc_biogpt_grl_benchmark/reports
current_crc_biogpt_grl_benchmark/tests
current_crc_biogpt_grl_benchmark/configs
current_crc_biogpt_grl_benchmark/data_manifest
```

Recommended reports to read first:

```text
current_crc_biogpt_grl_benchmark/reports/full551_benchmark_reproduction_summary.md
current_crc_biogpt_grl_benchmark/reports/full551_biogpt_cls_summary.md
current_crc_biogpt_grl_benchmark/reports/mmuphin_guided_residual_grl_full551_summary.md
current_crc_biogpt_grl_benchmark/reports/full551_grl_abundance_summary.md
current_crc_biogpt_grl_benchmark/reports/crc389_overlap_audit_vs_full551.md
```

### `scgpt_local_changes/`

This contains local scGPT-related modifications and minimal pipeline work. It is not a full copy of upstream scGPT.

The upstream scGPT repository was:

```text
https://github.com/bowang-lab/scGPT.git
```

This folder includes:

- `scgpt_local_changes.patch`;
- copies of locally modified tracked files;
- a minimal atlas pipeline;
- a smoke example;
- a cleaned Colab notebook.

This folder is included to document how scGPT ideas were used as a conceptual reference.

### `desktop_biogpt_selected/`

This contains selected lightweight files from the older Desktop Ali lab BiomeGPT workspace.

It includes:

- BiomeGPT full pipeline notebook;
- taxonomy BiomeGPT workflow notebook;
- BiomeGPT reproducibility training script;
- taxonomy pipeline script;
- professor smoke outputs;
- batch annotation phase 2 summaries.

This folder documents earlier BiomeGPT training and reproducibility scaffolds.

### `legacy_biogpt_batch_work/`

This contains earlier BiomeGPT batch-correction / batch-aware modeling work.

It includes:

- batch adversarial correction;
- centroid distillation;
- batch-conditioned decoder;
- batch-token pretraining;
- real-study embedding correction;
- legacy reports.

This is historical reference material. It should not be treated as the current canonical MMUPHin CRC benchmark.

### `top_level_crc_scripts/`

This contains R/Python scripts copied from the current workspace root, including:

```text
prepare_crc_controlled_benchmark.R
evaluate_crc_method.R
crc_benchmark_utils.R
validate_crc_benchmark.R
mmuphin_crc_scouting.R
crc_overlap_check.py
```

These scripts are related to MMUPHin CRC dataset preparation, overlap checking, and controlled evaluation.

## 3. What Is In The Teams Zip

Teams zip path:

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\teams_handoff_ali_lab_biogpt_scgpt_20260706.zip
```

The Teams zip is a broader handoff package. It contains more than code.

### `github_ready_code_20260706.zip`

This is the compressed version of the GitHub-ready code package.

If someone does not want to clone GitHub, they can unzip this file and inspect the code locally.

### `lab_handoff_biogpt_batch_correction_20260706.zip`

This is the current CRC/MMUPHin/BiomeGPT benchmark handoff package.

It includes:

- the current CRC/MMUPHin benchmark package;
- reports;
- figures;
- output summaries;
- the current handoff README.

### `LAB_HANDOFF_README_20260706.md`

This is the detailed technical handoff README.

It explains:

- project goal;
- current reliable conclusions;
- main reports;
- canonical benchmark metrics;
- BiomeGPT CLS results;
- what not to overclaim;
- recommended next scientific direction.

### `WHERE_TO_FIND_FILES_20260706.md`

This document.

It is included so that people can quickly understand what is in GitHub, what is in the Teams zip, and what was intentionally excluded.

### `selected_reports/current_crc_mmuphin/`

Current MMUPHin CRC / full551 professor-facing reports.

Main files:

```text
crc_batch_correction_progress_report.pdf
crc_batch_correction_progress_report.tex
mmuphin_crc_professor_report.pdf
mmuphin_crc_professor_report.md
full551_raw_mmuphin_pca.png
full551_old_grl_pca.png
full551_residual_grl_pca.png
```

The most useful professor-facing PDF is:

```text
crc_batch_correction_progress_report.pdf
```

### `selected_reports/desktop_legacy_reports/`

Selected older professor-facing PDFs and summary CSV/MD files.

These are useful for reconstructing the project history, but they are not the current most reliable benchmark.

### `optional_legacy_zips/`

Selected manageable historical zip packages, such as:

```text
biomegpt_reusable_20260521_batch_correction.zip
professor_batch_effect_minimal_repro_package.zip
professor_batch_effect_final_package.zip
biomegpt_vscode_ssh_package_20260520_124147.zip
```

These are optional references and do not need to be read first.

### `project_context/`

Small context files from the older working folders:

```text
ALI_LAB_PROJECT_MAP.md
shared_memory/project_memory.md
shared_memory/scgpt_biogpt_bridge.md
shared_memory/paths.json
```

These explain how the older `scgpt` and `biogpt` folders related to each other.

## 4. What Was Intentionally Excluded

The following were intentionally excluded from GitHub and from the main Teams handoff package:

- PyTorch checkpoints;
- raw data folders;
- 2GB/3GB historical zip archives;
- local virtual environments;
- `.git` folders;
- `.codex` / `.agents`;
- `tmp`;
- Python cache folders;
- LaTeX intermediate files;
- rendered PDF page images.

Checkpoint files not included:

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\taxonomy_checkpoint_stage1 (1).pt
C:\Users\Yuanshu\Documents\new_attemp_batch\taxonomy_checkpoint_stage2 (1).pt
```

If someone needs to rerun BiomeGPT CLS extraction, these checkpoints should be shared separately.

Large historical zip files not included:

```text
C:\Users\Yuanshu\Desktop\Ali lab\biomegpt_full_work_20260521_batch_correction.zip
C:\Users\Yuanshu\Desktop\Ali lab\outputs_phase2_start_scgpt_style_batch_aware (2).zip
C:\Users\Yuanshu\Desktop\Ali lab\outputs_phase2_start_scgpt_style_batch_aware.zip
```

These are too large for GitHub and not appropriate for a normal Teams handoff.

## 5. Safe Scientific Summary

Safe summary:

> We organized a controlled MMUPHin CRC benchmark and a BiomeGPT/scGPT-style batch-correction prototype. The current reliable finding is that BiomeGPT CLS embeddings contain measurable study/cohort signal, and simple post-hoc GRL can reduce study classifier predictability but does not reliably remove global study-associated variance. The stronger future direction is batch-aware BiomeGPT adapter/LoRA post-training with invariant/nuisance splitting, study-conditioned reconstruction, conditional CORAL/MMD, and strict LOSO/cross-fitted evaluation.

Do not claim:

- that the current GRL method definitively beats MMUPHin;
- that old cw0.1 is a clean biological improvement;
- that CRC389 is the canonical benchmark;
- that current CLS correction is already a final foundation-model-level batch correction method.

## 6. Updating GitHub Later

Local GitHub repo:

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\github_ready_code_20260706
```

Useful commands:

```powershell
cd "C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\github_ready_code_20260706"
git status
git add .
git commit -m "Update handoff documentation"
git push
```

Remote GitHub repository:

```text
https://github.com/yuanshu777/biomegpt-batch-correction-benchmark
```

## 7. Recommended Reading Order

If someone has only 10 minutes:

1. GitHub repo README;
2. `LAB_HANDOFF_README_20260706.md`;
3. `crc_batch_correction_progress_report.pdf`;
4. `full551_benchmark_reproduction_summary.md`;
5. `mmuphin_guided_residual_grl_full551_summary.md`.

If someone wants to continue development:

1. clone the GitHub repo;
2. inspect `current_crc_biogpt_grl_benchmark/src`;
3. inspect `current_crc_biogpt_grl_benchmark/scripts`;
4. request the checkpoint files separately if CLS extraction needs to be rerun;
5. avoid starting from old legacy outputs.

