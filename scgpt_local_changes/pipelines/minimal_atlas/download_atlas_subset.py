#!/usr/bin/env python
"""
Download a minimal CellxGene Census subset for scGPT experiments.

This script targets a practical MVP flow:
1) Query normal human cells by tissue (or custom value filter)
2) Randomly sample target number of cells
3) Download sampled cells in h5ad partitions
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import cellxgene_census
import numpy as np


MAJOR_TISSUES = ["heart", "blood", "brain", "lung", "kidney", "intestine", "pancreas"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a sampled CellxGene Atlas subset in h5ad partitions."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for index files, h5ad partitions and manifest.",
    )
    parser.add_argument(
        "--query-name",
        type=str,
        default="lung",
        help=(
            "Built-in query name. Supports major tissues plus 'others' and "
            "'all-normal'. Ignored when --value-filter is set."
        ),
    )
    parser.add_argument(
        "--value-filter",
        type=str,
        default=None,
        help="Custom Census value_filter expression. If set, overrides --query-name.",
    )
    parser.add_argument(
        "--organism",
        type=str,
        default="Homo sapiens",
        help="Organism string for Census get_anndata (default: Homo sapiens).",
    )
    parser.add_argument(
        "--organism-key",
        type=str,
        default="homo_sapiens",
        help="Organism key in Census store (default: homo_sapiens).",
    )
    parser.add_argument(
        "--census-version",
        type=str,
        default="stable",
        help="CellxGene Census release version (e.g. stable, latest, or a pinned date).",
    )
    parser.add_argument(
        "--target-cells",
        type=int,
        default=1_500_000,
        help="Target number of cells to sample.",
    )
    parser.add_argument(
        "--partition-size",
        type=int,
        default=50_000,
        help="Max number of cells per h5ad partition.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip partition files that already exist.",
    )
    parser.add_argument(
        "--max-partitions",
        type=int,
        default=None,
        help="For smoke tests: cap number of downloaded partitions.",
    )
    return parser.parse_args()


def build_value_filter(query_name: str) -> str:
    if query_name in MAJOR_TISSUES:
        return (
            "suspension_type != 'na' and disease == 'normal' and "
            f"tissue_general == '{query_name}'"
        )
    if query_name == "others":
        base = "suspension_type != 'na' and disease == 'normal'"
        for tissue in MAJOR_TISSUES:
            base += f" and tissue_general != '{tissue}'"
        return base
    if query_name == "all-normal":
        return "suspension_type != 'na' and disease == 'normal'"
    raise ValueError(
        f"Unknown query_name '{query_name}'. "
        f"Choose one of {MAJOR_TISSUES + ['others', 'all-normal']} "
        "or pass --value-filter."
    )


def write_idx(path: Path, idx: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in idx:
            f.write(f"{int(x)}\n")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    index_dir = output_dir / "index"
    h5ad_dir = output_dir / "h5ad"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    h5ad_dir.mkdir(parents=True, exist_ok=True)

    value_filter = args.value_filter or build_value_filter(args.query_name)
    print(f"[1/4] Querying soma ids from Census {args.census_version} ...")
    print(f"       value_filter: {value_filter}")

    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        obs = census["census_data"][args.organism_key].obs.read(
            value_filter=value_filter,
            column_names=["soma_joinid"],
        )
        obs_pd = obs.concat().to_pandas()

    all_ids = obs_pd["soma_joinid"].to_numpy(dtype=np.int64)
    n_total = int(all_ids.shape[0])
    print(f"       matched cells: {n_total:,}")
    if n_total == 0:
        raise RuntimeError("No cells matched the query. Adjust --query-name/--value-filter.")

    full_idx_path = index_dir / f"{args.query_name}.full.idx"
    write_idx(full_idx_path, all_ids)
    print(f"       full index saved: {full_idx_path}")

    n_sample = min(args.target_cells, n_total)
    if n_sample < args.target_cells:
        print(
            f"       warning: target_cells={args.target_cells:,} > matched cells; "
            f"using n_sample={n_sample:,}"
        )

    print(f"[2/4] Sampling {n_sample:,} cells (seed={args.seed}) ...")
    rng = np.random.default_rng(args.seed)
    if n_sample == n_total:
        sampled_ids = all_ids
    else:
        sampled_ids = rng.choice(all_ids, size=n_sample, replace=False)

    sampled_idx_path = index_dir / f"{args.query_name}.sampled_n{n_sample}.idx"
    write_idx(sampled_idx_path, sampled_ids)
    print(f"       sampled index saved: {sampled_idx_path}")

    n_partitions = math.ceil(n_sample / args.partition_size)
    if args.max_partitions is not None:
        n_partitions = min(n_partitions, args.max_partitions)
    print(
        "[3/4] Downloading partitions "
        f"(partition_size={args.partition_size:,}, total_partitions={n_partitions}) ..."
    )

    downloaded = 0
    skipped = 0
    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        for pidx in range(n_partitions):
            start = pidx * args.partition_size
            end = min((pidx + 1) * args.partition_size, n_sample)
            part_ids = sampled_ids[start:end]
            part_file = h5ad_dir / f"partition_{pidx:05d}.h5ad"
            if args.resume and part_file.exists():
                skipped += 1
                print(f"       [{pidx + 1}/{n_partitions}] skip existing {part_file.name}")
                continue

            print(
                f"       [{pidx + 1}/{n_partitions}] download {part_file.name} "
                f"({end - start:,} cells)"
            )
            adata = cellxgene_census.get_anndata(
                census,
                organism=args.organism,
                obs_coords=part_ids.tolist(),
            )
            adata.write_h5ad(part_file, compression="gzip")
            downloaded += 1

    manifest = {
        "query_name": args.query_name,
        "value_filter": value_filter,
        "organism": args.organism,
        "organism_key": args.organism_key,
        "census_version": args.census_version,
        "seed": args.seed,
        "target_cells": args.target_cells,
        "sampled_cells": n_sample,
        "matched_cells": n_total,
        "partition_size": args.partition_size,
        "partitions_written": downloaded,
        "partitions_skipped": skipped,
        "full_index_file": str(full_idx_path),
        "sampled_index_file": str(sampled_idx_path),
        "h5ad_dir": str(h5ad_dir),
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[4/4] Done. Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
