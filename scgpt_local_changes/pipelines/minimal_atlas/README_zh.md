# scGPT 最小可执行 Pipeline（Atlas 子集 1-2M cells）

这个目录提供一个可直接跑的 MVP 流程，目标是把 `scGPT` 在 CellxGene Atlas 子集上从 0 跑通。

流程包含 3 步：

1. 从 CellxGene Census 下载子集并分块为 `h5ad`
2. 把 `h5ad` 转为 scGPT 的 `scBank` 格式
3. （可选）用 scGPT checkpoint 计算 cell embedding

## 0. 环境前置

当前仓库依赖（`scanpy/scvi/cellxgene-census`）不建议在 Python 3.14 直接安装。  
建议使用 Python 3.10 或 3.11 的独立环境。

示例（你可按自己方式创建环境）：

```powershell
# 示例: 进入仓库根目录
cd "C:\Users\Yuanshu\Documents\ali lab\scgpt"

# 假设你已有 python3.10/3.11 可执行文件
py -3.11 -m venv .venv311
.venv311\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e .
pip install -r .\pipelines\minimal_atlas\requirements_mvp.txt
```

## 1. 一键跑 MVP

在仓库根目录执行：

```powershell
python .\pipelines\minimal_atlas\run_mvp_pipeline.py `
  --work-dir .\runs\atlas_mvp_lung_1p5m `
  --query-name lung `
  --census-version stable `
  --target-cells 1500000 `
  --partition-size 50000 `
  --resume-download `
  --skip-embed
```

说明：

- 这条命令会完成下载 + scBank 构建；
- `--skip-embed` 表示先不跑模型推理（可先确认数据链路跑通）；
- 输出目录主要包括：
  - `runs/.../dataset/h5ad/`
  - `runs/.../scb/`

## 2. 启用 embedding（可选）

你需要准备一个 checkpoint 目录，至少包含：

- `best_model.pt`
- `args.json`
- `vocab.json`

然后执行：

```powershell
python .\pipelines\minimal_atlas\run_mvp_pipeline.py `
  --work-dir .\runs\atlas_mvp_lung_1p5m `
  --skip-download `
  --skip-scb `
  --model-dir "D:\models\scgpt\whole-human" `
  --embed-batch-size 128 `
  --embed-num-workers 0 `
  --embed-device cuda `
  --embed-skip-existing
```

输出在：

- `runs/.../embeddings/*.emb.h5ad`

## 3. 先做烟雾测试（推荐）

先跑一个极小规模确认环境没问题：

```powershell
python .\pipelines\minimal_atlas\run_mvp_pipeline.py `
  --work-dir .\runs\atlas_smoke `
  --query-name lung `
  --target-cells 20000 `
  --partition-size 5000 `
  --max-partitions 2 `
  --model-dir "D:\models\scgpt\whole-human" `
  --embed-max-files 1 `
  --embed-max-cells-per-file 2000
```

## 4. 常见参数

- `--query-name`: `heart|blood|brain|lung|kidney|intestine|pancreas|others|all-normal`
- `--value-filter`: 直接传 Census 过滤表达式（会覆盖 `--query-name`）
- `--resume-download`: 下载断点续跑（跳过已存在分块）
- `--gene-min-count-n`: 传给 `build_large_scale_data.py` 的基因过滤超参

## 5. 关键脚本

- `download_atlas_subset.py`: Atlas 抽样 + 分块下载
- `embed_partitions_sparse.py`: 稀疏友好 embedding，避免整矩阵 densify
- `run_mvp_pipeline.py`: 三步串联入口
- `requirements_mvp.txt`: MVP 附加依赖清单
