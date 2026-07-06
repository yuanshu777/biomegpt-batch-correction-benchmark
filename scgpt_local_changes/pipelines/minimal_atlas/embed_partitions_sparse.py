#!/usr/bin/env python
"""
Generate scGPT embeddings for partitioned h5ad files in a sparse-friendly way.

Compared with the built-in embed_data utility, this script avoids converting the
entire sparse matrix to dense in memory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from anndata import AnnData
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained


class SparseCellDataset(Dataset):
    def __init__(
        self,
        X,
        gene_ids: np.ndarray,
        cls_id: int,
        pad_value: float,
    ) -> None:
        self.X = X.tocsr() if sp.issparse(X) else np.asarray(X)
        self.gene_ids = gene_ids
        self.cls_id = cls_id
        self.pad_value = float(pad_value)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        if sp.issparse(self.X):
            row = self.X.getrow(idx)
            nonzero_cols = row.indices
            values = row.data.astype(np.float32, copy=False)
        else:
            row = np.asarray(self.X[idx]).ravel()
            nonzero_cols = np.flatnonzero(row)
            values = row[nonzero_cols].astype(np.float32, copy=False)

        genes = self.gene_ids[nonzero_cols].astype(np.int64, copy=False)

        genes = np.concatenate(([self.cls_id], genes))
        values = np.concatenate(([self.pad_value], values))

        return {
            "id": idx,
            "genes": torch.from_numpy(genes),
            "expressions": torch.from_numpy(values),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed h5ad partitions with scGPT model checkpoint."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Input h5ad directory.")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for embedding h5ad files."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing best_model.pt, args.json, vocab.json.",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="partition_*.h5ad",
        help="Glob pattern to select input files.",
    )
    parser.add_argument(
        "--gene-col",
        type=str,
        default="auto",
        help="Gene column in adata.var. Use 'auto' to infer from feature_name/gene_name/index.",
    )
    parser.add_argument(
        "--obs-cols",
        type=str,
        default="cell_type,tissue,tissue_general,disease",
        help="Comma separated obs columns to keep in output if present.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=1200,
        help="Max sequence length after token collation.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu. Default uses cuda if available.",
    )
    parser.add_argument(
        "--use-fast-transformer",
        action="store_true",
        help="Enable fast transformer backend (flash-attn) when available.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="For smoke tests: only embed first N files.",
    )
    parser.add_argument(
        "--max-cells-per-file",
        type=int,
        default=None,
        help="For smoke tests: cap number of cells in each input file.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output files that already exist.",
    )
    return parser.parse_args()


def infer_gene_col(adata: AnnData, gene_col: str) -> str:
    if gene_col != "auto":
        if gene_col == "index":
            adata.var["index"] = adata.var.index
            return "index"
        if gene_col not in adata.var.columns:
            raise ValueError(f"gene_col '{gene_col}' not found in adata.var")
        return gene_col

    for candidate in ("feature_name", "gene_name"):
        if candidate in adata.var.columns:
            return candidate
    adata.var["index"] = adata.var.index
    return "index"


def load_model(model_dir: Path, device: torch.device, use_fast_transformer: bool):
    vocab_file = model_dir / "vocab.json"
    config_file = model_dir / "args.json"
    ckpt_file = model_dir / "best_model.pt"
    for p in (vocab_file, config_file, ckpt_file):
        if not p.exists():
            raise FileNotFoundError(f"Missing required model file: {p}")

    vocab = GeneVocab.from_file(vocab_file)
    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab["<pad>"])

    with config_file.open("r", encoding="utf-8") as f:
        model_configs = json.load(f)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    ckpt = torch.load(ckpt_file, map_location=device)
    load_pretrained(model, ckpt, verbose=False)
    model.to(device)
    model.eval()
    return model, vocab, model_configs


def embed_one_file(
    input_file: Path,
    output_file: Path,
    model,
    vocab: GeneVocab,
    model_configs: dict,
    gene_col: str,
    obs_cols: List[str],
    batch_size: int,
    max_length: int,
    num_workers: int,
    device: torch.device,
    max_cells_per_file: Optional[int],
) -> None:
    adata = sc.read_h5ad(input_file)
    if max_cells_per_file is not None and adata.n_obs > max_cells_per_file:
        adata = adata[:max_cells_per_file].copy()

    used_gene_col = infer_gene_col(adata, gene_col)
    adata.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in adata.var[used_gene_col]]
    adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()

    gene_names = adata.var[used_gene_col].tolist()
    gene_ids = np.array(vocab(gene_names), dtype=np.int64)

    dataset = SparseCellDataset(
        adata.X,
        gene_ids=gene_ids,
        cls_id=vocab["<cls>"],
        pad_value=float(model_configs["pad_value"]),
    )
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[model_configs["pad_token"]],
        pad_value=model_configs["pad_value"],
        do_mlm=False,
        do_binning=True,
        max_length=max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    embeddings = np.zeros((len(dataset), model_configs["embsize"]), dtype=np.float32)
    cursor = 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        for batch in tqdm(data_loader, desc=f"Embedding {input_file.name}"):
            input_gene_ids = batch["gene"].to(device)
            input_values = batch["expr"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[model_configs["pad_token"]])
            batch_emb = model._encode(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
            )[:, 0, :]
            batch_emb = batch_emb.detach().cpu().numpy().astype(np.float32, copy=False)
            embeddings[cursor : cursor + batch_emb.shape[0]] = batch_emb
            cursor += batch_emb.shape[0]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    embeddings = embeddings / norms

    keep_cols = [c for c in obs_cols if c in adata.obs.columns]
    obs_df = adata.obs[keep_cols].copy() if keep_cols else adata.obs.iloc[:, 0:0].copy()
    out_adata = sc.AnnData(X=embeddings, obs=obs_df, dtype=np.float32)
    out_adata.obs_names = adata.obs_names.copy()
    out_adata.uns["source_file"] = str(input_file)
    out_adata.uns["source_n_cells"] = int(adata.n_obs)
    out_adata.write_h5ad(output_file, compression="gzip")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model, vocab, model_configs = load_model(
        model_dir, device=device, use_fast_transformer=args.use_fast_transformer
    )

    files = sorted(input_dir.glob(args.glob_pattern))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir} with pattern {args.glob_pattern}")

    obs_cols = [x.strip() for x in args.obs_cols.split(",") if x.strip()]
    for in_file in files:
        out_file = output_dir / f"{in_file.stem}.emb.h5ad"
        if args.skip_existing and out_file.exists():
            print(f"Skip existing {out_file.name}")
            continue
        print(f"Embedding {in_file.name} -> {out_file.name}")
        embed_one_file(
            input_file=in_file,
            output_file=out_file,
            model=model,
            vocab=vocab,
            model_configs=model_configs,
            gene_col=args.gene_col,
            obs_cols=obs_cols,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
            device=device,
            max_cells_per_file=args.max_cells_per_file,
        )

    print(f"Done. Embeddings saved to {output_dir}")


if __name__ == "__main__":
    main()
