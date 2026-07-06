# 文件位置与分享包说明

日期：2026-07-06

这个文档说明本次整理后的 BiomeGPT / scGPT / MMUPHin CRC batch-correction 相关文件放在哪里，以及 GitHub repo 和 Teams zip 包里分别包含什么。

## 1. 最重要的两个分享入口

### GitHub repo

GitHub 地址：

```text
https://github.com/yuanshu777/biomegpt-batch-correction-benchmark
```

用途：

- 放代码为主；
- 方便别人在线浏览；
- 方便后续继续开发；
- 不放大数据、checkpoint、虚拟环境和巨型历史 zip。

本地对应文件夹：

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\github_ready_code_20260706
```

本地 GitHub-ready 压缩包：

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\github_ready_code_20260706.zip
```

当前已经推送到 GitHub 的 commit：

```text
7513340 Add BiomeGPT batch-correction handoff code
```

### Teams 分享包

Teams 推荐上传这个 zip：

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\teams_handoff_ali_lab_biogpt_scgpt_20260706.zip
```

用途：

- 给 lab / professor 直接下载；
- 包含 GitHub-ready code zip；
- 包含当前 CRC/MMUPHin benchmark handoff zip；
- 包含 professor-facing PDFs；
- 包含精选旧报告和可管理大小的 legacy package；
- 不包含 2GB/3GB 级别巨型历史 zip，也不包含 PyTorch checkpoint。

Teams zip 解压后的本地 staging 文件夹：

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\teams_handoff_20260706
```

## 2. GitHub repo 里面有什么

GitHub repo 是代码导向的整理版，主要包含以下几个目录。

### `current_crc_biogpt_grl_benchmark/`

这是当前最重要、最干净的项目代码。

内容包括：

- MMUPHin CRC benchmark 加载与整理；
- BiomeGPT checkpoint / CLS extraction 相关代码；
- GRL correction module；
- NoMean adapter；
- split/invariant-nuisance adapter；
- evaluation metrics；
- CRC389 / full551 scripts；
- full551 benchmark reports；
- smoke tests。

主要子目录：

```text
current_crc_biogpt_grl_benchmark/src
current_crc_biogpt_grl_benchmark/scripts
current_crc_biogpt_grl_benchmark/reports
current_crc_biogpt_grl_benchmark/tests
current_crc_biogpt_grl_benchmark/configs
current_crc_biogpt_grl_benchmark/data_manifest
```

最值得先看的报告：

```text
current_crc_biogpt_grl_benchmark/reports/full551_benchmark_reproduction_summary.md
current_crc_biogpt_grl_benchmark/reports/full551_biogpt_cls_summary.md
current_crc_biogpt_grl_benchmark/reports/mmuphin_guided_residual_grl_full551_summary.md
current_crc_biogpt_grl_benchmark/reports/full551_grl_abundance_summary.md
current_crc_biogpt_grl_benchmark/reports/crc389_overlap_audit_vs_full551.md
```

### `scgpt_local_changes/`

这是本地 scGPT 相关改动，不是完整复制上游 scGPT。

原始上游 repo 是：

```text
https://github.com/bowang-lab/scGPT.git
```

这里主要放：

- 本地 patch：`scgpt_local_changes.patch`
- 修改过的 tracked 文件副本；
- minimal atlas pipeline；
- smoke example；
- cleaned Colab notebook。

这个目录用于说明我们如何借鉴 scGPT 的思路，但不要把它当成完整 scGPT package。

### `desktop_biogpt_selected/`

这是从旧的 Desktop Ali lab BiomeGPT 工作区挑出来的轻量文件。

包含：

- BiomeGPT full pipeline notebook；
- taxonomy BiomeGPT workflow notebook；
- BiomeGPT reproducibility training script；
- taxonomy pipeline script；
- professor smoke outputs；
- batch annotation phase2 summaries。

这个目录说明早期 BiomeGPT 训练/复现 scaffold 是怎么做的。

### `legacy_biogpt_batch_work/`

这是更早的 BiomeGPT batch-correction / batch-aware modeling 工作。

包含：

- batch adversarial correction；
- centroid distillation；
- batch-conditioned decoder；
- batch-token pretraining；
- real-study embedding correction；
- legacy reports。

注意：这个目录是历史参考，不是当前 canonical MMUPHin CRC benchmark。

### `top_level_crc_scripts/`

这里放从当前 workspace 顶层整理出来的 R/Python 脚本，例如：

```text
prepare_crc_controlled_benchmark.R
evaluate_crc_method.R
crc_benchmark_utils.R
validate_crc_benchmark.R
mmuphin_crc_scouting.R
crc_overlap_check.py
```

这些脚本和当前 CRC/MMUPHin benchmark 的数据准备与评估有关。

## 3. Teams zip 里面有什么

Teams zip 是更完整的交接包，不只是代码。

路径：

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\teams_handoff_ali_lab_biogpt_scgpt_20260706.zip
```

里面主要包括：

### `github_ready_code_20260706.zip`

这是 GitHub 代码包的压缩版。

如果别人不想从 GitHub clone，也可以直接解压这个 zip 看代码。

### `lab_handoff_biogpt_batch_correction_20260706.zip`

这是当前 CRC/MMUPHin/BiomeGPT benchmark 的完整 handoff 包。

内容比 GitHub repo 更偏结果交接，包括：

- current CRC/MMUPHin benchmark package；
- reports；
- figures；
- output summaries；
- current handoff README。

### `LAB_HANDOFF_README_20260706.md`

这是最详细的技术交接 README。

它说明：

- project goal；
- current reliable conclusion；
- main reports；
- canonical benchmark metrics；
- BiomeGPT CLS results；
- what not to overclaim；
- next scientific direction。

### `selected_reports/current_crc_mmuphin/`

当前 MMUPHin CRC / full551 相关 professor-facing report。

主要包括：

```text
crc_batch_correction_progress_report.pdf
crc_batch_correction_progress_report.tex
mmuphin_crc_professor_report.pdf
mmuphin_crc_professor_report.md
full551_raw_mmuphin_pca.png
full551_old_grl_pca.png
full551_residual_grl_pca.png
```

其中最推荐直接发给 professor 看的 PDF 是：

```text
crc_batch_correction_progress_report.pdf
```

### `selected_reports/desktop_legacy_reports/`

这里放旧阶段的 professor-facing PDFs 和 summary CSV/MD。

这些文件用于说明之前一步步做过的 batch-effect / BiomeGPT / scGPT-style exploration。

注意：这些是历史探索材料，不是当前最可靠的 final benchmark。

### `optional_legacy_zips/`

这里放几个大小可管理的历史包，例如：

```text
biomegpt_reusable_20260521_batch_correction.zip
professor_batch_effect_minimal_repro_package.zip
professor_batch_effect_final_package.zip
biomegpt_vscode_ssh_package_20260520_124147.zip
```

这些包是可选参考，不是必须先看的文件。

### `project_context/`

这里放旧工作区的 context / project map：

```text
ALI_LAB_PROJECT_MAP.md
shared_memory/project_memory.md
shared_memory/scgpt_biogpt_bridge.md
shared_memory/paths.json
```

用于解释旧的 `scgpt` 和 `biogpt` 文件夹之间的关系。

## 4. 没有放进去的东西

以下内容没有放进 GitHub，也没有放进 Teams 主包：

- PyTorch checkpoints；
- raw data 大文件；
- 2GB/3GB 历史 zip；
- local virtual environments；
- `.git` 文件夹；
- `.codex` / `.agents`；
- `tmp`；
- Python cache；
- LaTeX 中间文件；
- rendered PDF page images。

主要没放的 checkpoint：

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\taxonomy_checkpoint_stage1 (1).pt
C:\Users\Yuanshu\Documents\new_attemp_batch\taxonomy_checkpoint_stage2 (1).pt
```

如果别人需要重新抽 BiomeGPT CLS，需要单独分享这两个 checkpoint。

主要没放的超大历史 zip：

```text
C:\Users\Yuanshu\Desktop\Ali lab\biomegpt_full_work_20260521_batch_correction.zip
C:\Users\Yuanshu\Desktop\Ali lab\outputs_phase2_start_scgpt_style_batch_aware (2).zip
C:\Users\Yuanshu\Desktop\Ali lab\outputs_phase2_start_scgpt_style_batch_aware.zip
```

这些文件太大，不适合 GitHub，也不适合普通 Teams handoff。

## 5. 当前项目结论应该怎么说

安全说法：

> We organized a controlled MMUPHin CRC benchmark and a BiomeGPT/scGPT-style batch-correction prototype. The current reliable finding is that BiomeGPT CLS embeddings contain measurable study/cohort signal, and simple post-hoc GRL can reduce study classifier predictability but does not reliably remove global study-associated variance. The stronger future direction is batch-aware BiomeGPT adapter/LoRA post-training with invariant/nuisance splitting, study-conditioned reconstruction, conditional CORAL/MMD, and strict LOSO/cross-fitted evaluation.

不要说：

- 当前 GRL 已经全面打败 MMUPHin；
- old cw0.1 是 clean biological improvement；
- CRC389 是 canonical benchmark；
- 当前 CLS correction 已经是最终 foundation-model-level batch correction。

## 6. 如果之后继续更新 GitHub

本地 GitHub repo 位置：

```text
C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\github_ready_code_20260706
```

常用命令：

```powershell
cd "C:\Users\Yuanshu\Documents\new_attemp_batch\share_out\github_ready_code_20260706"
git status
git add .
git commit -m "Update handoff documentation"
git push
```

远程 GitHub：

```text
https://github.com/yuanshu777/biomegpt-batch-correction-benchmark
```

## 7. 最推荐给别人看的顺序

如果对方只有 10 分钟：

1. GitHub repo README；
2. `LAB_HANDOFF_README_20260706.md`；
3. `crc_batch_correction_progress_report.pdf`；
4. `full551_benchmark_reproduction_summary.md`；
5. `mmuphin_guided_residual_grl_full551_summary.md`。

如果对方要接着开发：

1. clone GitHub repo；
2. 看 `current_crc_biogpt_grl_benchmark/src`；
3. 看 `current_crc_biogpt_grl_benchmark/scripts`；
4. 如果要重新抽 CLS，单独拿 checkpoint；
5. 不要从 old legacy outputs 开始。

