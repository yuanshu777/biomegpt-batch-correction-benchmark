#!/usr/bin/env python
"""
Minimal scGPT finetune smoke test.

This script is adapted from examples/finetune_integration.py but intentionally
keeps only the smallest dependency surface and fastest runnable path:
- dataset: scanpy.datasets.pbmc3k()  (2,700 cells)
- no scvi-tools
- no wandb
- no scib metrics

Goal: verify end-to-end finetune/training pipeline can run before scaling.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scanpy as sc
import torch
import torchtext
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Silence torchtext deprecation warning spam in logs.
torchtext.disable_torchtext_deprecation_warning()
from torchtext._torchtext import Vocab as VocabPybind
from torchtext.vocab import Vocab

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scgpt as scg
from scgpt.loss import masked_mse_loss, masked_relative_error
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import GeneVocab, random_mask_value, tokenize_and_pad_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal scGPT finetune smoke test.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "runs" / "finetune_smoke_pbmc3k"),
        help="Directory to save logs/checkpoints.",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Optional checkpoint dir containing best_model.pt, args.json, vocab.json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cells", type=int, default=2500)
    parser.add_argument("--n-hvg", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--layer-size", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--n-layers-cls", type=int, default=3)
    parser.add_argument("--schedule-ratio", type=float, default=0.9)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    parser.add_argument(
        "--use-fast-transformer",
        action="store_true",
        help="Enable fast transformer backend if available.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self) -> int:
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.data.items()}


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset=SeqDataset(data_pt),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.output_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    logger.info("Starting minimal finetune smoke run")
    logger.info(f"Args: {vars(args)}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # 1) Load tiny public dataset from scanpy
    adata = sc.datasets.pbmc3k()  # 2700 cells x 32738 genes
    if args.max_cells > 0 and adata.n_obs > args.max_cells:
        adata = adata[: args.max_cells].copy()
    adata.var["gene_name"] = adata.var_names.astype(str)
    logger.info(f"Loaded dataset shape: {adata.shape}")

    # 2) Optional vocab/model loading
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_value = -1
    pad_value = -2
    n_input_bins = None
    load_model = args.load_model is not None

    if load_model:
        model_dir = Path(args.load_model).resolve()
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"
        for p in (model_config_file, model_file, vocab_file):
            if not p.exists():
                raise FileNotFoundError(f"Missing required file: {p}")

        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        with model_config_file.open("r", encoding="utf-8") as f:
            model_configs = json.load(f)
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]

        adata.var["id_in_vocab"] = [1 if g in vocab else -1 for g in adata.var["gene_name"]]
        in_vocab = np.array(adata.var["id_in_vocab"])
        logger.info(f"Genes in pretrained vocab: {int(np.sum(in_vocab >= 0))}/{len(in_vocab)}")
        adata = adata[:, in_vocab >= 0].copy()
    else:
        embsize = args.layer_size
        nhead = args.nhead
        d_hid = args.layer_size
        nlayers = args.nlayers
        n_layers_cls = args.n_layers_cls
        vocab = None
        model_file = None

    # 3) Preprocess and bin
    n_hvg = min(args.n_hvg, adata.n_vars)
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=3,
        filter_cell_by_counts=1,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=n_hvg,
        hvg_flavor="cell_ranger",
        binning=None,
    )
    preprocessor(adata)
    logger.info(f"Post-preprocess shape: {adata.shape}, n_hvg={n_hvg}")

    input_layer_key = "X_log1p"
    all_counts = adata.layers[input_layer_key].toarray() if issparse(adata.layers[input_layer_key]) else adata.layers[input_layer_key]
    genes = adata.var["gene_name"].tolist()

    train_data, valid_data = train_test_split(
        all_counts,
        test_size=0.1,
        shuffle=True,
        random_state=args.seed,
    )
    logger.info(f"Split train/valid: {train_data.shape[0]}/{valid_data.shape[0]}")

    if vocab is None:
        vocab = Vocab(VocabPybind(genes + special_tokens, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    max_seq_len = n_hvg + 1
    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )

    logger.info(
        "Tokenized train shape=%s, valid shape=%s",
        tuple(tokenized_train["genes"].shape),
        tuple(tokenized_valid["genes"].shape),
    )

    def prepare_data() -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        masked_values_train = random_mask_value(
            tokenized_train["values"],
            mask_ratio=args.mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        masked_values_valid = random_mask_value(
            tokenized_valid["values"],
            mask_ratio=args.mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        train_pt = {
            "gene_ids": tokenized_train["genes"],
            "values": masked_values_train,
            "target_values": tokenized_train["values"],
        }
        valid_pt = {
            "gene_ids": tokenized_valid["genes"],
            "values": masked_values_valid,
            "target_values": tokenized_valid["values"],
        }
        return train_pt, valid_pt

    # 4) Build model (lightweight path: MLM only)
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=embsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=args.dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        n_input_bins=n_input_bins,
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=args.use_fast_transformer,
        pre_norm=False,
    )
    if model_file is not None:
        pretrained_dict = torch.load(model_file, map_location=device)
        model_dict = model.state_dict()
        filtered = {
            k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        logger.info(
            f"Loaded pretrained params: {len(filtered)}/{len(model_dict)} tensors"
        )

    model.to(device)
    criterion = masked_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.schedule_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    def train_epoch(epoch: int, loader: DataLoader) -> None:
        model.train()
        total_loss = 0.0
        total_error = 0.0
        start = time.time()
        for step, batch in enumerate(loader, 1):
            input_gene_ids = batch["gene_ids"].to(device)
            input_values = batch["values"].to(device)
            target_values = batch["target_values"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    MVC=False,
                    ECS=False,
                )
                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_dict["mlm_output"], target_values, masked_positions)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                mre = masked_relative_error(output_dict["mlm_output"], target_values, masked_positions)
            total_loss += loss.item()
            total_error += mre.item()
            if step % args.log_interval == 0 or step == len(loader):
                logger.info(
                    f"epoch={epoch} step={step}/{len(loader)} "
                    f"loss={total_loss/step:.4f} mre={total_error/step:.4f}"
                )
        logger.info(f"epoch={epoch} train_time={time.time()-start:.2f}s")

    @torch.no_grad()
    def evaluate(loader: DataLoader) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_error = 0.0
        total_num = 0
        for batch in loader:
            input_gene_ids = batch["gene_ids"].to(device)
            input_values = batch["values"].to(device)
            target_values = batch["target_values"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    MVC=False,
                    ECS=False,
                )
                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_dict["mlm_output"], target_values, masked_positions)
            bsz = input_gene_ids.shape[0]
            total_loss += loss.item() * bsz
            total_error += (
                masked_relative_error(output_dict["mlm_output"], target_values, masked_positions).item() * bsz
            )
            total_num += bsz
        return total_loss / max(total_num, 1), total_error / max(total_num, 1)

    # 5) Train
    best_val = float("inf")
    best_model = None
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_pt, valid_pt = prepare_data()
        train_loader = prepare_dataloader(train_pt, batch_size=args.batch_size, shuffle=True)
        valid_loader = prepare_dataloader(valid_pt, batch_size=args.batch_size, shuffle=False)

        train_epoch(epoch, train_loader)
        val_loss, val_mre = evaluate(valid_loader)
        logger.info(f"epoch={epoch} valid_loss={val_loss:.4f} valid_mre={val_mre:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            logger.info(f"New best model at epoch={epoch}, valid_loss={best_val:.4f}")

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), save_dir / f"model_e{epoch}.pt")

        scheduler.step()

    if best_model is None:
        best_model = model
        best_epoch = args.epochs

    # 6) Save artifacts
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    if hasattr(vocab, "save_json"):
        vocab.save_json(save_dir / "vocab.json")
    run_args = vars(args).copy()
    run_args.update(
        {
            "pad_token": pad_token,
            "pad_value": pad_value,
            "embsize": embsize,
            "nheads": nhead,
            "d_hid": d_hid,
            "nlayers": nlayers,
            "n_layers_cls": n_layers_cls,
            "dropout": args.dropout,
        }
    )
    with (save_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(run_args, f, indent=2)

    logger.info(
        f"Done. best_epoch={best_epoch}, best_valid_loss={best_val:.4f}, output={save_dir}"
    )


if __name__ == "__main__":
    main()
