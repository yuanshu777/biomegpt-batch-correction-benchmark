"""
Distill cross-fitted real-study mean correction into the BiomeGPT encoder.

The earlier post-hoc baseline showed that subtracting cross-fitted study means from
sample-prompt embeddings can almost eliminate real-study predictability. This script
turns that successful representation-level correction into a model-level training
objective:

    total loss = masked-abundance reconstruction loss
               + target_weight * MSE(cls_embedding, corrected_target_embedding)

The target embedding is computed from the base checkpoint embeddings and real study ids.
The correction is cross-fitted so a sample's target does not use its own embedding in the
study mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from batch_adversarial_correction import BalancedStudyBatchSampler, select_label_panel
from batch_effect_diagnostics import keep_labels_with_min_count, make_hd_labels, multiclass_probe
from biomegpt_taxonomy_pipeline import (
    abundance_to_binned_matrix,
    build_mask,
    extract_sample_embeddings,
    load_checkpoint_model,
    load_csv_or_zip,
)
from real_study_embedding_correction import crossfit_study_center


def prepare_training_data(
    args: argparse.Namespace,
    species: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    abund = load_csv_or_zip(Path(args.data_dir) / "abund_pretraining_phase2_gut.csv.zip")
    meta = pd.read_csv(args.batch_annotation_csv, low_memory=False).set_index("sample_id")
    meta = meta.loc[meta.index.intersection(abund.index)].copy()
    abund = abund.loc[meta.index]

    panel_mask = select_label_panel(meta, args)
    meta = meta.loc[panel_mask].copy()
    counts = meta["batch_label_external_recommended"].astype(str).value_counts()
    keep = meta["batch_label_external_recommended"].astype(str).map(counts).ge(args.min_batch_size)
    meta = meta.loc[keep].copy()
    abund = abund.loc[meta.index].reindex(columns=species, fill_value=0.0)

    label_text = meta["batch_label_external_recommended"].astype(str)
    labels = LabelEncoder().fit_transform(label_text).astype(np.int64)

    payload = np.load(args.before_embeddings_npz, allow_pickle=True)
    before_ids = payload["sample_ids"].astype(str)
    before = pd.DataFrame(payload["embeddings"].astype(np.float32), index=before_ids)
    base_embeddings = before.loc[meta.index].to_numpy(dtype=np.float32)
    targets, _folds = crossfit_study_center(
        base_embeddings,
        label_text.to_numpy(),
        args.n_splits,
        args.seed,
        args.target_method,
        args.eps,
    )
    return abund, meta, labels, targets.astype(np.float32)


def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rng: torch.Generator,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(args.mixed_precision and device.type == "cuda"))
    losses: List[float] = []
    recon_losses: List[float] = []
    target_losses: List[float] = []

    for bins, _labels, target in loader:
        bins = bins.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = build_mask(bins, args.mask_ratio, rng)
        valid = mask & (bins > 0)
        if valid.sum().item() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            pred = model.forward_pretrain(bins, mask)
            recon_loss = F.mse_loss(pred[valid], bins[valid].float())
            cls = model.sample_prompt(bins).float()
            target_float = target.float()
            if args.target_normalize:
                cls = F.layer_norm(cls, (cls.shape[-1],))
                target_float = F.layer_norm(target_float, (target_float.shape[-1],))
            target_loss = F.mse_loss(cls, target_float)
            target_weight = float(getattr(args, "_effective_target_weight", args.target_weight))
            loss = recon_loss + target_weight * target_loss

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        recon_losses.append(float(recon_loss.detach().cpu()))
        target_losses.append(float(target_loss.detach().cpu()))

    return {
        "loss": float(np.mean(losses)),
        "reconstruction_loss": float(np.mean(recon_losses)),
        "target_loss": float(np.mean(target_losses)),
    }


def extract_all_phase2_embeddings(args: argparse.Namespace, checkpoint: Path, out_npz: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    device = torch.device(args.device)
    model, species, _payload = load_checkpoint_model(checkpoint, Path(args.taxonomy_xlsx), device)
    abund = load_csv_or_zip(Path(args.data_dir) / "abund_pretraining_phase2_gut.csv.zip")
    meta = pd.read_csv(args.batch_annotation_csv, low_memory=False).set_index("sample_id")
    abund = abund.loc[abund.index.intersection(meta.index)]
    meta = meta.loc[abund.index].copy()
    abund = abund.reindex(columns=species, fill_value=0.0)
    bins = abundance_to_binned_matrix(abund, args.bins)
    emb = extract_sample_embeddings(model, bins, device, args.embedding_batch_size)
    np.savez_compressed(out_npz, embeddings=emb, sample_ids=abund.index.to_numpy(), species=np.array(species))
    return emb, meta


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, species, base_payload = load_checkpoint_model(Path(args.checkpoint), Path(args.taxonomy_xlsx), device)
    abund, train_meta, labels, targets = prepare_training_data(args, species)
    bins = abundance_to_binned_matrix(abund, args.bins)
    dataset = TensorDataset(torch.from_numpy(bins).long(), torch.from_numpy(labels).long(), torch.from_numpy(targets).float())
    if args.balanced_study_batches:
        sampler = BalancedStudyBatchSampler(labels, args.studies_per_batch, args.samples_per_study, args.seed)
        loader = DataLoader(dataset, batch_sampler=sampler, pin_memory=(device.type == "cuda"))
    else:
        loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator(device=device.type if device.type in {"cpu", "cuda"} else "cpu").manual_seed(args.seed)

    history = []
    for epoch in range(1, args.epochs + 1):
        scale = min(1.0, epoch / args.target_warmup_epochs) if args.target_warmup_epochs > 0 else 1.0
        args._effective_target_weight = args.target_weight * scale
        row = train_epoch(model, loader, optimizer, device, rng, args)
        row["epoch"] = epoch
        row["effective_target_weight"] = float(args._effective_target_weight)
        history.append(row)
        print(
            f"[distill] epoch {epoch:03d}/{args.epochs} "
            f"loss={row['loss']:.4f} recon={row['reconstruction_loss']:.4f} "
            f"target={row['target_loss']:.4f} target_w={row['effective_target_weight']:.4f}"
        )

    corrected_ckpt = out_dir / f"taxonomy_checkpoint_stage2_real_study_{args.label_panel}_{args.target_method}_distill.pt"
    payload = dict(base_payload)
    payload["model_state_dict"] = model.state_dict()
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["extra"] = dict(payload.get("extra", {}))
    payload["extra"]["real_study_centroid_distillation"] = {
        "history": history,
        "label_panel": args.label_panel,
        "target_method": args.target_method,
        "n_training_samples": int(len(train_meta)),
        "n_batch_labels": int(np.max(labels) + 1),
        "args": vars(args),
    }
    torch.save(payload, corrected_ckpt)

    before_payload = np.load(args.before_embeddings_npz, allow_pickle=True)
    before_emb = before_payload["embeddings"]
    before_ids = before_payload["sample_ids"].astype(str)
    phase2_meta = pd.read_csv(args.batch_annotation_csv, low_memory=False).set_index("sample_id").loc[before_ids].copy()
    phenotype_col = "Phenotype_fullname" if "Phenotype_fullname" in phase2_meta.columns else "Phenotype"
    phase2_meta["hd_label"] = make_hd_labels(phase2_meta[phenotype_col])
    panel_mask = select_label_panel(phase2_meta, args).to_numpy()
    panel_counts = phase2_meta.loc[panel_mask, "batch_label_external_recommended"].astype(str).value_counts()
    batch_eval_mask = panel_mask & phase2_meta["batch_label_external_recommended"].astype(str).map(panel_counts).ge(args.min_batch_size).to_numpy()

    after_emb, after_meta = extract_all_phase2_embeddings(
        args,
        corrected_ckpt,
        out_dir / "phase2_sample_prompt_embeddings_after_centroid_distillation.npz",
    )
    after_meta["hd_label"] = make_hd_labels(after_meta[phenotype_col])

    probes = [
        multiclass_probe(
            before_emb[batch_eval_mask],
            phase2_meta.loc[batch_eval_mask, "batch_label_external_recommended"],
            f"{args.label_panel}_batch_before",
            args.seed,
            args.test_size,
        ),
        multiclass_probe(
            after_emb[batch_eval_mask],
            after_meta.loc[batch_eval_mask, "batch_label_external_recommended"],
            f"{args.label_panel}_batch_after",
            args.seed,
            args.test_size,
        ),
        multiclass_probe(before_emb, phase2_meta["hd_label"], "hd_before", args.seed, args.test_size),
        multiclass_probe(after_emb, after_meta["hd_label"], "hd_after", args.seed, args.test_size),
    ]
    summary = {
        "corrected_checkpoint": str(corrected_ckpt),
        "label_panel": args.label_panel,
        "target_method": args.target_method,
        "training_history": history,
        "training_samples": int(len(train_meta)),
        "batch_classes": int(np.max(labels) + 1),
        "before_after_probe_results": probes,
        "interpretation": {
            "desired_direction": "batch_after macro-F1 should decrease while H/D and reconstruction remain usable.",
            "method": "Distill cross-fitted real-study mean-corrected base embeddings into the model cls output.",
        },
    }
    with open(out_dir / "centroid_distillation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(history).to_csv(out_dir / "centroid_distillation_training_history.csv", index=False)
    pd.DataFrame(probes).to_csv(out_dir / "before_after_probe_metrics.csv", index=False)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill real-study centroid-corrected embeddings into BiomeGPT")
    p.add_argument("--data_dir", default="dataset_v3")
    p.add_argument("--taxonomy_xlsx", default="dataset_v3/species_taxonomy_filled_validated_Serena.xlsx")
    p.add_argument("--checkpoint", default="dataset_v3/outputs_taxonomy_notebook/taxonomy_checkpoint_stage2.pt")
    p.add_argument("--before_embeddings_npz", default="dataset_v3/outputs_batch_diagnostics_real_study/phase2_sample_prompt_embeddings.npz")
    p.add_argument("--batch_annotation_csv", default="dataset_v3/meta_pretraining_phase2_gut_real_study_annotation.csv")
    p.add_argument("--output_dir", default="dataset_v3/outputs_batch_centroid_distillation_real_study")
    p.add_argument("--label_panel", choices=["conservative_safe", "high_only", "high_medium"], default="conservative_safe")
    p.add_argument("--target_method", choices=["mean_center", "mean_scale"], default="mean_center")
    p.add_argument("--target_weight", type=float, default=1.0)
    p.add_argument("--target_warmup_epochs", type=float, default=0.0)
    p.add_argument("--target_normalize", action="store_true")
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--embedding_batch_size", type=int, default=64)
    p.add_argument("--min_batch_size", type=int, default=10)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--balanced_study_batches", action="store_true")
    p.add_argument("--studies_per_batch", type=int, default=8)
    p.add_argument("--samples_per_study", type=int, default=4)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
