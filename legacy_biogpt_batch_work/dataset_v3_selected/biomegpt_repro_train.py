"""
Minimal BiomeGPT-style species pretraining script for dataset_v3.

What this script does:
1) Reports mean/median/25th/75th percentiles of non-zero species count per sample
   for the gut + non-gut pretraining dataset.
2) Runs two-stage pretraining:
   - Stage 1: gut + non-gut dataset for 30 epochs (default), with 32 bins (default).
   - Stage 2: domain adaptation on gut-only dataset for additional epochs (default 5).

Notes:
- This is a practical reproduction scaffold based on the paper description.
- It is not guaranteed to be bitwise identical to the authors' internal code.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]
    TORCH_AVAILABLE = False


@dataclass
class NonZeroStats:
    mean: float
    median: float
    p25: float
    p75: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_csv_from_path(path: Path, index_col: int = 0) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            members = [
                m
                for m in zf.namelist()
                if not m.endswith("/") and not m.startswith("__MACOSX/")
            ]
            if not members:
                raise RuntimeError(f"No CSV file found inside zip: {path}")
            # Prefer first CSV-looking file if multiple exist
            members = sorted(members, key=lambda x: (not x.lower().endswith(".csv"), x))
            with zf.open(members[0]) as f:
                return pd.read_csv(f, index_col=index_col)
    return pd.read_csv(path, index_col=index_col)


def align_abund_meta(abund: pd.DataFrame, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = abund.index.intersection(meta.index)
    if len(common) == 0:
        raise RuntimeError("No overlapping sample IDs between abundance and metadata.")
    return abund.loc[common], meta.loc[common]


def compute_nonzero_stats(abund: pd.DataFrame) -> NonZeroStats:
    nonzero_counts = (abund > 0).sum(axis=1)
    return NonZeroStats(
        mean=float(nonzero_counts.mean()),
        median=float(nonzero_counts.median()),
        p25=float(nonzero_counts.quantile(0.25)),
        p75=float(nonzero_counts.quantile(0.75)),
    )


def print_stats(label: str, stats: NonZeroStats) -> None:
    print(f"\n[{label}] non-zero species per sample:")
    print(f"  Mean:   {stats.mean:.4f}")
    print(f"  Median: {stats.median:.4f}")
    print(f"  P25:    {stats.p25:.4f}")
    print(f"  P75:    {stats.p75:.4f}")


def rank_to_bins(nonzero_values: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Convert non-zero abundances to rank-based bins in [1, num_bins].

    - Highest abundance gets highest bin.
    - If N >= bins: evenly partition ranks into bins.
    - If N < bins: spread values across full 1..bins range.
    """
    n = nonzero_values.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(-nonzero_values)  # descending by abundance
    bins = np.zeros(n, dtype=np.int64)

    if n >= num_bins:
        # rank 0 -> bin B, rank n-1 -> bin 1
        ranks = np.arange(n, dtype=np.float64)
        assigned = num_bins - np.floor(ranks * num_bins / n).astype(np.int64)
        assigned = np.clip(assigned, 1, num_bins)
    else:
        # spread across full [1, B]
        if n == 1:
            assigned = np.array([num_bins], dtype=np.int64)
        else:
            ranks = np.arange(n, dtype=np.float64)
            assigned = np.round((n - 1 - ranks) * (num_bins - 1) / (n - 1)).astype(np.int64) + 1
            assigned = np.clip(assigned, 1, num_bins)

    bins[order] = assigned
    return bins


def abundance_to_binned_matrix(abund: pd.DataFrame, num_bins: int) -> np.ndarray:
    arr = abund.to_numpy(dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.int64)
    for i in range(arr.shape[0]):
        row = arr[i]
        nz_idx = np.flatnonzero(row > 0)
        if nz_idx.size == 0:
            continue
        nz_vals = row[nz_idx]
        out[i, nz_idx] = rank_to_bins(nz_vals, num_bins)
    return out


if TORCH_AVAILABLE:
    class SpeciesBinDataset(Dataset):
        def __init__(self, binned_matrix: np.ndarray):
            self.x = torch.from_numpy(binned_matrix).long()

        def __len__(self) -> int:
            return self.x.shape[0]

        def __getitem__(self, idx: int) -> torch.Tensor:
            return self.x[idx]


    class BiomeGPTSpeciesModel(nn.Module):
        def __init__(
            self,
            num_species: int,
            num_bins: int,
            d_model: int = 512,
            nhead: int = 8,
            num_layers: int = 8,
            ff_dim: int = 512,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.num_species = num_species
            self.num_bins = num_bins
            self.d_model = d_model
            self.nhead = nhead

            # species ids: 0 = <cls>, 1..num_species = species tokens
            self.species_emb = nn.Embedding(num_species + 1, d_model)
            self.abund_mlp = nn.Sequential(
                nn.Linear(1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.ln_species = nn.LayerNorm(d_model)
            self.ln_abund = nn.LayerNorm(d_model)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                activation="relu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

            self.head = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1),  # regress masked abundance-bin value
            )

            species_ids = torch.arange(1, num_species + 1, dtype=torch.long).unsqueeze(0)
            self.register_buffer("species_ids", species_ids, persistent=False)

        def _build_paper_attention_mask(self, bins: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """
            Build per-sample attention mask to match paper behavior:
            - zero-abundance species do not participate in cross-token attention
            - unmasked species attend to unmasked species
            - masked species attend to unmasked species and themselves
            """
            bsz, seq_len = bins.shape
            device = bins.device
            l = seq_len + 1  # +1 for <cls>

            nonzero = bins > 0
            unmasked = nonzero & (~mask)
            masked = nonzero & mask

            allow = torch.zeros((bsz, l, l), dtype=torch.bool, device=device)

            # <cls> attends to itself and unmasked species.
            allow[:, 0, 0] = True
            allow[:, 0, 1:] = unmasked

            # Species-to-species attention rules.
            unmasked_q = unmasked.unsqueeze(2)  # [B, S, 1]
            masked_q = masked.unsqueeze(2)      # [B, S, 1]
            unmasked_k = unmasked.unsqueeze(1)  # [B, 1, S]

            species_allow = (unmasked_q & unmasked_k) | (masked_q & unmasked_k)

            # Always allow self-attention for each species token so rows are never fully masked.
            eye = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)  # [1, S, S]
            species_allow = species_allow | eye

            # Restrict species->species block.
            allow[:, 1:, 1:] = species_allow

            # MultiheadAttention expects bool mask: True means "disallow attention".
            disallow = ~allow  # [B, L, L]
            disallow = (
                disallow.unsqueeze(1)
                .expand(bsz, self.nhead, l, l)
                .reshape(bsz * self.nhead, l, l)
            )
            return disallow

        def forward(self, bins: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            bins: [B, S], integer bin labels 0..num_bins
            mask: [B, S], bool for masked positions
            """
            bsz, seq_len = bins.shape
            if seq_len != self.num_species:
                raise ValueError(f"Expected {self.num_species} species, got {seq_len}")

            input_bins = bins.clone()
            input_bins[mask] = 0

            cls_id = torch.zeros((bsz, 1), dtype=torch.long, device=bins.device)
            species_ids = self.species_ids.expand(bsz, -1)
            all_species_ids = torch.cat([cls_id, species_ids], dim=1)  # [B, S+1]

            cls_bin = torch.zeros((bsz, 1), dtype=torch.long, device=bins.device)
            all_bins = torch.cat([cls_bin, input_bins], dim=1)  # [B, S+1]

            species_tok = self.ln_species(self.species_emb(all_species_ids))
            abund_input = (all_bins.float() / max(self.num_bins, 1)).unsqueeze(-1)
            abund_tok = self.ln_abund(self.abund_mlp(abund_input))
            x = species_tok + abund_tok

            attn_mask = self._build_paper_attention_mask(bins=bins, mask=mask)
            h = self.encoder(x, mask=attn_mask)  # [B, S+1, D]
            preds = self.head(h[:, 1:, :]).squeeze(-1)  # [B, S]
            return preds, input_bins


    def build_mask(batch_bins: torch.Tensor, mask_ratio: float, rng: torch.Generator) -> torch.Tensor:
        """
        Mask ~mask_ratio of non-zero tokens for each sample.
        """
        bsz, seq_len = batch_bins.shape
        mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=batch_bins.device)
        for i in range(bsz):
            nz_idx = torch.nonzero(batch_bins[i] > 0, as_tuple=False).squeeze(-1)
            if nz_idx.numel() == 0:
                continue
            k = max(1, int(round(mask_ratio * nz_idx.numel())))
            perm = torch.randperm(nz_idx.numel(), generator=rng, device=batch_bins.device)
            chosen = nz_idx[perm[:k]]
            mask[i, chosen] = True
        return mask


    def train_stage(
        model: BiomeGPTSpeciesModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int,
        mask_ratio: float,
        stage_name: str,
        seed: int,
        mixed_precision: bool,
    ) -> None:
        scaler = torch.amp.GradScaler("cuda", enabled=(mixed_precision and device.type == "cuda"))
        rng = torch.Generator(device=device.type if device.type in {"cpu", "cuda"} else "cpu")
        rng.manual_seed(seed)

        model.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            steps = 0

            for batch_bins in loader:
                batch_bins = batch_bins.to(device, non_blocking=True)
                mask = build_mask(batch_bins, mask_ratio=mask_ratio, rng=rng)
                valid = mask & (batch_bins > 0)
                if valid.sum().item() == 0:
                    continue

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(mixed_precision and device.type == "cuda"),
                ):
                    preds, _ = model(batch_bins, mask)
                    loss = F.mse_loss(preds[valid], batch_bins[valid].float())

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                running_loss += float(loss.detach().cpu().item())
                steps += 1

            epoch_loss = running_loss / max(steps, 1)
            print(f"[{stage_name}] epoch {epoch:02d}/{epochs} - loss: {epoch_loss:.6f}")


    def save_checkpoint(
        path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, args: argparse.Namespace
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(payload, path)
        print(f"Saved checkpoint: {path}")

else:
    class SpeciesBinDataset:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("PyTorch is required for training.")

    class BiomeGPTSpeciesModel:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("PyTorch is required for training.")

    def train_stage(*_args, **_kwargs) -> None:  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for training.")

    def save_checkpoint(*_args, **_kwargs) -> None:  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for training.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BiomeGPT minimal reproduction trainer")
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--phase1_abund", type=str, default="abund_pretraining_phase1_gut_and_nongut.csv.zip")
    p.add_argument("--phase1_meta", type=str, default="meta_pretraining_phase1_gut_and_nongut.csv")
    p.add_argument("--phase2_abund", type=str, default="abund_pretraining_phase2_gut.csv.zip")
    p.add_argument("--phase2_meta", type=str, default="meta_pretraining_phase2_gut.csv")
    p.add_argument("--output_dir", type=str, default="outputs_biomegpt_repro")

    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--epochs_phase1", type=int, default=30)
    p.add_argument("--epochs_phase2", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--num_workers", type=int, default=0)
    default_device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    p.add_argument("--device", type=str, default=default_device)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--report_only", action="store_true")
    p.add_argument("--cache_binned", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load phase1 data (gut + non-gut) ----
    phase1_abund = load_csv_from_path(data_dir / args.phase1_abund, index_col=0)
    phase1_meta = load_csv_from_path(data_dir / args.phase1_meta, index_col=0)
    phase1_abund, phase1_meta = align_abund_meta(phase1_abund, phase1_meta)

    # Teacher-required stats
    phase1_stats = compute_nonzero_stats(phase1_abund)
    print_stats("Phase1 gut+non-gut", phase1_stats)

    stats_path = output_dir / "phase1_nonzero_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(asdict(phase1_stats), f, indent=2)
    print(f"Saved stats to: {stats_path}")

    if args.report_only:
        return

    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install torch first, or run with --report_only for stats."
        )

    # ---- Load phase2 data (gut only) ----
    phase2_abund = load_csv_from_path(data_dir / args.phase2_abund, index_col=0)
    phase2_meta = load_csv_from_path(data_dir / args.phase2_meta, index_col=0)
    phase2_abund, phase2_meta = align_abund_meta(phase2_abund, phase2_meta)

    # Keep phase1 species vocabulary for continuity between stage1 and stage2.
    phase1_species = phase1_abund.columns.tolist()
    phase2_abund = phase2_abund.reindex(columns=phase1_species, fill_value=0.0)

    # ---- Convert abundance -> binned matrix ----
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    phase1_cache = cache_dir / f"phase1_bins{args.bins}.npy"
    phase2_cache = cache_dir / f"phase2_bins{args.bins}.npy"

    if args.cache_binned and phase1_cache.exists():
        phase1_bins = np.load(phase1_cache)
    else:
        phase1_bins = abundance_to_binned_matrix(phase1_abund, num_bins=args.bins)
        if args.cache_binned:
            np.save(phase1_cache, phase1_bins)
    if args.cache_binned and phase2_cache.exists():
        phase2_bins = np.load(phase2_cache)
    else:
        phase2_bins = abundance_to_binned_matrix(phase2_abund, num_bins=args.bins)
        if args.cache_binned:
            np.save(phase2_cache, phase2_bins)

    # ---- Build model and loaders ----
    device = torch.device(args.device)
    num_species = phase1_bins.shape[1]
    model = BiomeGPTSpeciesModel(
        num_species=num_species,
        num_bins=args.bins,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    phase1_loader = DataLoader(
        SpeciesBinDataset(phase1_bins),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    phase2_loader = DataLoader(
        SpeciesBinDataset(phase2_bins),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ---- Stage 1: gut + non-gut pretraining ----
    train_stage(
        model=model,
        loader=phase1_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs_phase1,
        mask_ratio=args.mask_ratio,
        stage_name="Stage1 pretrain (gut+non-gut)",
        seed=args.seed,
        mixed_precision=args.mixed_precision,
    )
    save_checkpoint(output_dir / "checkpoint_stage1.pt", model, optimizer, args)

    # ---- Stage 2: gut-only domain adaptation ----
    train_stage(
        model=model,
        loader=phase2_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs_phase2,
        mask_ratio=args.mask_ratio,
        stage_name="Stage2 domain-adapt (gut-only)",
        seed=args.seed + 1,
        mixed_precision=args.mixed_precision,
    )
    save_checkpoint(output_dir / "checkpoint_stage2.pt", model, optimizer, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
