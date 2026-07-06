#!/usr/bin/env python
"""
Run the minimal executable scGPT pipeline on a CellxGene Atlas subset:
1) Download sampled h5ad partitions
2) Build scBank files
3) (Optional) Compute scGPT embeddings
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run minimal scGPT pipeline from 0 to runnable outputs."
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="Root working directory for pipeline outputs.",
    )

    # Step 1: download args
    parser.add_argument("--query-name", type=str, default="lung")
    parser.add_argument("--value-filter", type=str, default=None)
    parser.add_argument("--census-version", type=str, default="stable")
    parser.add_argument("--target-cells", type=int, default=1_500_000)
    parser.add_argument("--partition-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-download", action="store_true")
    parser.add_argument("--max-partitions", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")

    # Step 2: build scb args
    parser.add_argument("--skip-scb", action="store_true")
    parser.add_argument(
        "--vocab-file",
        type=str,
        default=None,
        help="Vocab file for scBank build. Default uses scgpt/tokenizer/default_census_vocab.json",
    )
    parser.add_argument(
        "--gene-min-count-n",
        type=int,
        default=200_000,
        help="N hyperparameter passed to build_large_scale_data.py",
    )

    # Step 3: embedding args
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="If provided, run embedding step with this checkpoint folder.",
    )
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--embed-max-length", type=int, default=1200)
    parser.add_argument("--embed-num-workers", type=int, default=0)
    parser.add_argument("--embed-max-files", type=int, default=None)
    parser.add_argument("--embed-max-cells-per-file", type=int, default=None)
    parser.add_argument("--embed-device", type=str, default="cuda")
    parser.add_argument("--embed-skip-existing", action="store_true")

    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = work_dir / "dataset"
    h5ad_dir = dataset_dir / "h5ad"
    scb_dir = work_dir / "scb"
    emb_dir = work_dir / "embeddings"

    download_script = repo_root / "pipelines" / "minimal_atlas" / "download_atlas_subset.py"
    build_scb_script = repo_root / "data" / "cellxgene" / "build_large_scale_data.py"
    embed_script = repo_root / "pipelines" / "minimal_atlas" / "embed_partitions_sparse.py"

    if args.vocab_file is None:
        vocab_file = repo_root / "scgpt" / "tokenizer" / "default_census_vocab.json"
    else:
        vocab_file = Path(args.vocab_file).resolve()

    # Step 1: download data
    if not args.skip_download:
        cmd = [
            sys.executable,
            str(download_script),
            "--output-dir",
            str(dataset_dir),
            "--query-name",
            args.query_name,
            "--census-version",
            args.census_version,
            "--target-cells",
            str(args.target_cells),
            "--partition-size",
            str(args.partition_size),
            "--seed",
            str(args.seed),
        ]
        if args.value_filter:
            cmd.extend(["--value-filter", args.value_filter])
        if args.resume_download:
            cmd.append("--resume")
        if args.max_partitions is not None:
            cmd.extend(["--max-partitions", str(args.max_partitions)])
        run(cmd, cwd=repo_root)
    else:
        print("Skip step 1 (download).")

    # Step 2: build scb
    if not args.skip_scb:
        cmd = [
            sys.executable,
            str(build_scb_script),
            "--input-dir",
            str(h5ad_dir),
            "--output-dir",
            str(scb_dir),
            "--vocab-file",
            str(vocab_file),
            "--N",
            str(args.gene_min_count_n),
        ]
        run(cmd, cwd=repo_root / "data" / "cellxgene")
    else:
        print("Skip step 2 (scb build).")

    # Step 3: embeddings
    if not args.skip_embed and args.model_dir:
        cmd = [
            sys.executable,
            str(embed_script),
            "--input-dir",
            str(h5ad_dir),
            "--output-dir",
            str(emb_dir),
            "--model-dir",
            str(Path(args.model_dir).resolve()),
            "--batch-size",
            str(args.embed_batch_size),
            "--max-length",
            str(args.embed_max_length),
            "--num-workers",
            str(args.embed_num_workers),
            "--device",
            args.embed_device,
        ]
        if args.embed_max_files is not None:
            cmd.extend(["--max-files", str(args.embed_max_files)])
        if args.embed_max_cells_per_file is not None:
            cmd.extend(["--max-cells-per-file", str(args.embed_max_cells_per_file)])
        if args.embed_skip_existing:
            cmd.append("--skip-existing")
        run(cmd, cwd=repo_root)
    else:
        print("Skip step 3 (embedding). Provide --model-dir to enable.")

    print("Pipeline completed.")
    print(f"dataset_dir: {dataset_dir}")
    print(f"h5ad_dir:    {h5ad_dir}")
    print(f"scb_dir:     {scb_dir}")
    print(f"emb_dir:     {emb_dir}")


if __name__ == "__main__":
    main()
