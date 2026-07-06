"""
Masked-abundance reconstruction diagnostics for batch-correction experiments.

This evaluates the foundation-model objective directly. For each checkpoint, it masks the
same nonzero species positions and measures whether masked abundance-bin reconstruction is
preserved overall and across batch/study labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from batch_effect_diagnostics import str_to_bool_series
from biomegpt_taxonomy_pipeline import abundance_to_binned_matrix, load_checkpoint_model, load_csv_or_zip


def select_label_panel(meta: pd.DataFrame, panel: str) -> pd.Series:
    if panel == "conservative_safe":
        return str_to_bool_series(meta["safe_for_final_batch_correction_conservative"])
    if panel == "high_only":
        return meta["external_confidence"].astype(str).str.lower().eq("high")
    if panel == "high_medium":
        return meta["external_confidence"].astype(str).str.lower().isin(["high", "medium"])
    if panel == "all_labeled":
        return meta["batch_label_external_recommended"].notna()
    raise ValueError(f"Unknown label_panel: {panel}")


def make_deterministic_mask(bins: np.ndarray, mask_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros_like(bins, dtype=bool)
    for i in range(bins.shape[0]):
        nz = np.flatnonzero(bins[i] > 0)
        if nz.size == 0:
            continue
        k = max(1, int(round(mask_ratio * nz.size)))
        chosen = rng.choice(nz, size=min(k, nz.size), replace=False)
        mask[i, chosen] = True
    return mask


def evaluate_checkpoint(
    checkpoint: Path,
    taxonomy_xlsx: Path,
    bins: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int,
    mixed_precision: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model, _species, _payload = load_checkpoint_model(checkpoint, taxonomy_xlsx, device)
    model.eval()
    sample_sse = np.zeros(bins.shape[0], dtype=np.float64)
    sample_count = np.zeros(bins.shape[0], dtype=np.int64)

    scaler_enabled = mixed_precision and device.type == "cuda"
    with torch.no_grad():
        for start in range(0, bins.shape[0], batch_size):
            end = min(start + batch_size, bins.shape[0])
            batch_bins = torch.from_numpy(bins[start:end]).long().to(device)
            batch_mask = torch.from_numpy(mask[start:end]).bool().to(device)
            valid = batch_mask & (batch_bins > 0)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler_enabled):
                pred = model.forward_pretrain(batch_bins, batch_mask)
            err = (pred.float() - batch_bins.float()).pow(2)
            err = err.masked_fill(~valid, 0.0)
            sample_sse[start:end] = err.sum(dim=1).detach().cpu().numpy()
            sample_count[start:end] = valid.sum(dim=1).detach().cpu().numpy()

    total_count = int(sample_count.sum())
    overall_mse = float(sample_sse.sum() / total_count) if total_count else float("nan")
    return sample_sse, sample_count, overall_mse


def summarize_panel(
    meta: pd.DataFrame,
    sample_sse: np.ndarray,
    sample_count: np.ndarray,
    label_panel: str,
    min_batch_size: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    panel_mask = select_label_panel(meta, label_panel).to_numpy()
    labels = meta["batch_label_external_recommended"].astype(str)
    counts = labels[panel_mask].value_counts()
    eval_mask = panel_mask & labels.map(counts).ge(min_batch_size).to_numpy() & (sample_count > 0)

    panel_sse = float(sample_sse[eval_mask].sum())
    panel_count = int(sample_count[eval_mask].sum())
    panel_mse = panel_sse / panel_count if panel_count else float("nan")

    rows = []
    eval_df = pd.DataFrame(
        {
            "sample_pos": np.arange(len(meta), dtype=np.int64),
            "batch_label": labels.to_numpy(),
        },
        index=meta.index,
    ).loc[eval_mask]
    for label, group in eval_df.groupby("batch_label"):
        loc = group["sample_pos"].to_numpy(dtype=np.int64)
        sse = float(sample_sse[loc].sum())
        count = int(sample_count[loc].sum())
        rows.append(
            {
                "batch_label": label,
                "n_samples": int(len(loc)),
                "masked_positions": count,
                "mse": sse / count if count else float("nan"),
            }
        )
    per_batch = pd.DataFrame(rows).sort_values("mse", ascending=False)
    mse_values = per_batch["mse"].to_numpy(dtype=float) if len(per_batch) else np.array([], dtype=float)
    summary = {
        "label_panel": label_panel,
        "n_eval_samples": int(eval_mask.sum()),
        "n_batches": int(len(per_batch)),
        "panel_mse": float(panel_mse),
        "batch_mse_mean": float(np.mean(mse_values)) if mse_values.size else float("nan"),
        "batch_mse_std": float(np.std(mse_values)) if mse_values.size else float("nan"),
        "batch_mse_cv": float(np.std(mse_values) / np.mean(mse_values)) if mse_values.size and np.mean(mse_values) else float("nan"),
        "batch_mse_min": float(np.min(mse_values)) if mse_values.size else float("nan"),
        "batch_mse_max": float(np.max(mse_values)) if mse_values.size else float("nan"),
        "batch_mse_gap": float(np.max(mse_values) - np.min(mse_values)) if mse_values.size else float("nan"),
    }
    return summary, per_batch


def parse_checkpoint_arg(values: List[str]) -> List[Tuple[str, Path]]:
    out = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(value).stem
        out.append((name, Path(path)))
    return out


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoints = parse_checkpoint_arg(args.checkpoint)
    if not checkpoints:
        raise ValueError("At least one --checkpoint must be provided.")

    first_model, species, _payload = load_checkpoint_model(checkpoints[0][1], Path(args.taxonomy_xlsx), device)
    del first_model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    abund = load_csv_or_zip(Path(args.data_dir) / "abund_pretraining_phase2_gut.csv.zip")
    meta = pd.read_csv(args.batch_annotation_csv, low_memory=False).set_index("sample_id")
    sample_ids = abund.index.intersection(meta.index)
    abund = abund.loc[sample_ids].reindex(columns=species, fill_value=0.0)
    meta = meta.loc[sample_ids].copy()

    bins = abundance_to_binned_matrix(abund, args.bins)
    mask = make_deterministic_mask(bins, args.mask_ratio, args.seed)

    summaries = []
    for name, checkpoint in checkpoints:
        print(f"[recon] evaluating {name}: {checkpoint}")
        sample_sse, sample_count, overall_mse = evaluate_checkpoint(
            checkpoint,
            Path(args.taxonomy_xlsx),
            bins,
            mask,
            device,
            args.batch_size,
            args.mixed_precision,
        )
        base = {
            "checkpoint_name": name,
            "checkpoint": str(checkpoint),
            "overall_samples": int(len(meta)),
            "overall_masked_positions": int(sample_count.sum()),
            "overall_mse": float(overall_mse),
        }
        for panel in args.label_panel:
            panel_summary, per_batch = summarize_panel(meta, sample_sse, sample_count, panel, args.min_batch_size)
            row = {**base, **panel_summary}
            summaries.append(row)
            per_batch.to_csv(out_dir / f"{name}_{panel}_per_batch_reconstruction_mse.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "reconstruction_diagnostics_summary.csv", index=False)
    with open(out_dir / "reconstruction_diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summaries": summaries}, f, indent=2)
    print(summary_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate masked-abundance reconstruction across batch panels.")
    p.add_argument("--data_dir", default="dataset_v3")
    p.add_argument("--taxonomy_xlsx", default="dataset_v3/species_taxonomy_filled_validated_Serena.xlsx")
    p.add_argument("--batch_annotation_csv", default="dataset_v3/meta_pretraining_phase2_gut_batch_annotation_external_enriched.csv")
    p.add_argument("--output_dir", default="dataset_v3/outputs_batch_reconstruction_diagnostics")
    p.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint as name=path. Can be repeated.",
    )
    p.add_argument("--label_panel", action="append", default=None, choices=["conservative_safe", "high_only", "high_medium", "all_labeled"])
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument("--min_batch_size", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.label_panel is None:
        args.label_panel = ["conservative_safe"]
    return args


if __name__ == "__main__":
    run(parse_args())
