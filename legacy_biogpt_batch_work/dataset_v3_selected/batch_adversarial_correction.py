"""
Adversarial batch-correction trials for taxonomy-aware BiomeGPT.

This is a controlled follow-up to batch_effect_diagnostics.py. By default it
uses `safe_for_final_batch_correction_conservative=True` labels, but it can
also run high-only or high+medium label panels for ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, Sampler, TensorDataset
from sklearn.preprocessing import LabelEncoder

from batch_effect_diagnostics import make_hd_labels, multiclass_probe, str_to_bool_series
from biomegpt_taxonomy_pipeline import (
    abundance_to_binned_matrix,
    build_mask,
    extract_sample_embeddings,
    load_checkpoint_model,
    load_csv_or_zip,
)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class BatchDiscriminator(nn.Module):
    def __init__(self, d_model: int, n_batches: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_batches),
        )

    def forward(self, x: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        return self.net(grad_reverse(x, grl_lambda))


class BalancedStudyBatchSampler(Sampler[List[int]]):
    """Build minibatches with repeated samples per study for alignment losses."""

    def __init__(
        self,
        labels: np.ndarray,
        studies_per_batch: int,
        samples_per_study: int,
        seed: int,
    ):
        self.labels = np.asarray(labels)
        self.studies_per_batch = int(studies_per_batch)
        self.samples_per_study = int(samples_per_study)
        self.seed = int(seed)
        self.by_label = {
            int(label): np.flatnonzero(self.labels == label)
            for label in np.unique(self.labels)
        }
        self.labels_unique = np.array(sorted(self.by_label), dtype=np.int64)
        self.num_batches = int(np.ceil(len(self.labels) / max(1, self.studies_per_batch * self.samples_per_study)))
        self._iter_count = 0

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._iter_count)
        self._iter_count += 1
        for _ in range(self.num_batches):
            chosen_labels = rng.choice(
                self.labels_unique,
                size=min(self.studies_per_batch, len(self.labels_unique)),
                replace=False,
            )
            batch: List[int] = []
            for label in chosen_labels:
                candidates = self.by_label[int(label)]
                replace = len(candidates) < self.samples_per_study
                picked = rng.choice(candidates, size=self.samples_per_study, replace=replace)
                batch.extend(int(i) for i in picked)
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def select_label_panel(meta: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.label_panel == "conservative_safe":
        return str_to_bool_series(meta["safe_for_final_batch_correction_conservative"])
    if args.label_panel == "high_only":
        return meta["external_confidence"].astype(str).str.lower().eq("high")
    if args.label_panel == "high_medium":
        return meta["external_confidence"].astype(str).str.lower().isin(["high", "medium"])
    raise ValueError(f"Unknown label_panel: {args.label_panel}")


def prepare_batch_data(args: argparse.Namespace, species: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    data_dir = Path(args.data_dir)
    abund = load_csv_or_zip(data_dir / "abund_pretraining_phase2_gut.csv.zip")
    meta = pd.read_csv(args.batch_annotation_csv, low_memory=False).set_index("sample_id")
    meta = meta.loc[meta.index.intersection(abund.index)].copy()
    abund = abund.loc[meta.index]

    panel_mask = select_label_panel(meta, args)
    meta = meta.loc[panel_mask].copy()
    counts = meta["batch_label_external_recommended"].astype(str).value_counts()
    keep = meta["batch_label_external_recommended"].astype(str).map(counts).ge(args.min_batch_size)
    meta = meta.loc[keep].copy()
    abund = abund.loc[meta.index].reindex(columns=species, fill_value=0.0)

    labels = LabelEncoder().fit_transform(meta["batch_label_external_recommended"].astype(str))
    return abund, meta, labels.astype(np.int64)


def grouped_centroid_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    min_group_size: int,
) -> torch.Tensor:
    centroids = []
    for label in torch.unique(labels):
        idx = labels == label
        if int(idx.sum().item()) >= min_group_size:
            centroids.append(z[idx].mean(dim=0))
    if len(centroids) < 2:
        return z.new_tensor(0.0)
    c = torch.stack(centroids, dim=0)
    global_center = c.mean(dim=0, keepdim=True)
    return (c - global_center).pow(2).mean()


def covariance(x: torch.Tensor) -> torch.Tensor:
    centered = x - x.mean(dim=0, keepdim=True)
    denom = max(int(x.shape[0]) - 1, 1)
    return centered.T.matmul(centered) / denom


def grouped_coral_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    min_group_size: int,
) -> torch.Tensor:
    groups = []
    for label in torch.unique(labels):
        idx = labels == label
        if int(idx.sum().item()) >= min_group_size:
            groups.append(z[idx])
    if len(groups) < 2:
        return z.new_tensor(0.0)
    selected = torch.cat(groups, dim=0)
    global_cov = covariance(selected)
    d = z.shape[1]
    losses = [(covariance(group) - global_cov).pow(2).sum() / (4.0 * d * d) for group in groups]
    return torch.stack(losses).mean()


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, sigmas: Tuple[float, ...]) -> torch.Tensor:
    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    loss = x.new_tensor(0.0)
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma * sigma)
        loss = loss + torch.exp(-gamma * xx).mean() + torch.exp(-gamma * yy).mean() - 2.0 * torch.exp(-gamma * xy).mean()
    return loss / len(sigmas)


def grouped_mmd_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    min_group_size: int,
    sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    groups = []
    for label in torch.unique(labels):
        idx = labels == label
        if int(idx.sum().item()) >= min_group_size:
            groups.append(z[idx])
    if len(groups) < 2:
        return z.new_tensor(0.0)
    selected = torch.cat(groups, dim=0)
    losses = [rbf_mmd2(group, selected, sigmas) for group in groups]
    return torch.stack(losses).mean()


def representation_alignment_loss(
    cls: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.alignment_loss == "none":
        return cls.new_tensor(0.0)
    z = cls.float()
    if args.alignment_normalize:
        z = F.layer_norm(z, (z.shape[-1],))
    if args.alignment_loss == "centroid":
        return grouped_centroid_loss(z, labels, args.alignment_min_group_size)
    if args.alignment_loss == "coral":
        return grouped_coral_loss(z, labels, args.alignment_min_group_size)
    if args.alignment_loss == "mmd":
        return grouped_mmd_loss(z, labels, args.alignment_min_group_size)
    raise ValueError(f"Unknown alignment_loss: {args.alignment_loss}")


def train_adversarial_epoch(
    model,
    discriminator: BatchDiscriminator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rng: torch.Generator,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    discriminator.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(args.mixed_precision and device.type == "cuda"))
    losses: List[float] = []
    recon_losses: List[float] = []
    dab_losses: List[float] = []
    dab_accs: List[float] = []
    alignment_losses: List[float] = []

    for bins, batch_labels in loader:
        bins = bins.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        mask = build_mask(bins, args.mask_ratio, rng)
        valid = mask & (bins > 0)
        if valid.sum().item() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            h = model.encode(bins, mask=mask)
            pred = model.reconstruction_head(h[:, 1:, :]).squeeze(-1)
            recon_loss = F.mse_loss(pred[valid], bins[valid].float())
            grl_lambda = float(getattr(args, "_effective_grl_lambda", args.grl_lambda))
            dab_weight = float(getattr(args, "_effective_dab_weight", args.dab_weight))
            alignment_weight = float(getattr(args, "_effective_alignment_weight", args.alignment_weight))
            dab_logits = discriminator(h[:, 0, :], grl_lambda)
            dab_loss = F.cross_entropy(dab_logits, batch_labels)
            align_loss = representation_alignment_loss(h[:, 0, :], batch_labels, args)
            loss = recon_loss + dab_weight * dab_loss + alignment_weight * align_loss

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(discriminator.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(discriminator.parameters()), 1.0)
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        recon_losses.append(float(recon_loss.detach().cpu()))
        dab_losses.append(float(dab_loss.detach().cpu()))
        dab_accs.append(float((dab_logits.argmax(dim=1) == batch_labels).float().mean().detach().cpu()))
        alignment_losses.append(float(align_loss.detach().cpu()))

    return {
        "loss": float(np.mean(losses)),
        "reconstruction_loss": float(np.mean(recon_losses)),
        "dab_loss": float(np.mean(dab_losses)),
        "dab_train_accuracy": float(np.mean(dab_accs)),
        "alignment_loss": float(np.mean(alignment_losses)),
    }


def extract_all_phase2_embeddings(args: argparse.Namespace, checkpoint: Path, out_npz: Path) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
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
    return emb, meta, species


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, species, base_payload = load_checkpoint_model(Path(args.checkpoint), Path(args.taxonomy_xlsx), device)
    abund, train_meta, batch_labels = prepare_batch_data(args, species)
    bins = abundance_to_binned_matrix(abund, args.bins)
    dataset = TensorDataset(torch.from_numpy(bins).long(), torch.from_numpy(batch_labels).long())
    if args.balanced_study_batches:
        batch_sampler = BalancedStudyBatchSampler(
            batch_labels,
            args.studies_per_batch,
            args.samples_per_study,
            args.seed,
        )
        loader = DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=(device.type == "cuda"))
    else:
        loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=(device.type == "cuda"))

    discriminator = BatchDiscriminator(model.d_model, int(np.max(batch_labels)) + 1, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    rng = torch.Generator(device=device.type if device.type in {"cpu", "cuda"} else "cpu").manual_seed(args.seed)

    history = []
    for epoch in range(1, args.epochs + 1):
        dab_scale = min(1.0, epoch / args.dab_warmup_epochs) if args.dab_warmup_epochs > 0 else 1.0
        grl_scale = min(1.0, epoch / args.grl_warmup_epochs) if args.grl_warmup_epochs > 0 else 1.0
        alignment_scale = min(1.0, epoch / args.alignment_warmup_epochs) if args.alignment_warmup_epochs > 0 else 1.0
        args._effective_dab_weight = args.dab_weight * dab_scale
        args._effective_grl_lambda = args.grl_lambda * grl_scale
        args._effective_alignment_weight = args.alignment_weight * alignment_scale
        row = train_adversarial_epoch(model, discriminator, loader, optimizer, device, rng, args)
        row["epoch"] = epoch
        row["effective_dab_weight"] = float(args._effective_dab_weight)
        row["effective_grl_lambda"] = float(args._effective_grl_lambda)
        row["effective_alignment_weight"] = float(args._effective_alignment_weight)
        history.append(row)
        print(
            f"[adv] epoch {epoch:03d}/{args.epochs} "
            f"loss={row['loss']:.4f} recon={row['reconstruction_loss']:.4f} "
            f"dab={row['dab_loss']:.4f} dab_acc={row['dab_train_accuracy']:.4f} "
            f"align={row['alignment_loss']:.4f} "
            f"dab_w={row['effective_dab_weight']:.4f} grl={row['effective_grl_lambda']:.4f} "
            f"align_w={row['effective_alignment_weight']:.4f}"
        )

    suffix = args.label_panel if args.alignment_loss == "none" else f"{args.label_panel}_{args.alignment_loss}_align"
    corrected_ckpt = out_dir / f"taxonomy_checkpoint_stage2_batch_adversarial_{suffix}.pt"
    corrected_payload = dict(base_payload)
    corrected_payload["model_state_dict"] = model.state_dict()
    corrected_payload["optimizer_state_dict"] = optimizer.state_dict()
    corrected_payload["extra"] = dict(corrected_payload.get("extra", {}))
    corrected_payload["extra"]["batch_adversarial_correction"] = {
        "history": history,
        "label_panel": args.label_panel,
        "n_training_samples": int(len(train_meta)),
        "n_batch_labels": int(np.max(batch_labels) + 1),
        "args": vars(args),
        "discriminator_state_dict": discriminator.state_dict(),
    }
    torch.save(corrected_payload, corrected_ckpt)

    before_npz = Path(args.before_embeddings_npz)
    before_payload = np.load(before_npz, allow_pickle=True)
    before_emb = before_payload["embeddings"]
    before_ids = before_payload["sample_ids"].astype(str)
    phase2_meta = pd.read_csv(args.batch_annotation_csv, low_memory=False).set_index("sample_id").loc[before_ids].copy()
    phenotype_col = "Phenotype_fullname" if "Phenotype_fullname" in phase2_meta.columns else "Phenotype"
    phase2_meta["hd_label"] = make_hd_labels(phase2_meta[phenotype_col])
    panel_mask = select_label_panel(phase2_meta, args).to_numpy()
    panel_counts = phase2_meta.loc[panel_mask, "batch_label_external_recommended"].astype(str).value_counts()
    batch_eval_mask = panel_mask & phase2_meta["batch_label_external_recommended"].astype(str).map(panel_counts).ge(args.min_batch_size).to_numpy()

    after_emb, after_meta, _species = extract_all_phase2_embeddings(
        args,
        corrected_ckpt,
        out_dir / "phase2_sample_prompt_embeddings_after_batch_adversarial.npz",
    )
    after_meta["hd_label"] = make_hd_labels(after_meta[phenotype_col])

    before_batch_probe = multiclass_probe(
        before_emb[batch_eval_mask],
        phase2_meta.loc[batch_eval_mask, "batch_label_external_recommended"],
        f"{args.label_panel}_batch_before",
        args.seed,
        args.test_size,
    )
    after_batch_probe = multiclass_probe(
        after_emb[batch_eval_mask],
        after_meta.loc[batch_eval_mask, "batch_label_external_recommended"],
        f"{args.label_panel}_batch_after",
        args.seed,
        args.test_size,
    )
    before_hd_probe = multiclass_probe(before_emb, phase2_meta["hd_label"], "hd_before", args.seed, args.test_size)
    after_hd_probe = multiclass_probe(after_emb, after_meta["hd_label"], "hd_after", args.seed, args.test_size)

    summary = {
        "corrected_checkpoint": str(corrected_ckpt),
        "label_panel": args.label_panel,
        "training_history": history,
        "training_samples": int(len(train_meta)),
        "batch_classes": int(np.max(batch_labels) + 1),
        "before_after_probe_results": [
            before_batch_probe,
            after_batch_probe,
            before_hd_probe,
            after_hd_probe,
        ],
        "interpretation": {
            "desired_direction": "batch_after balanced_accuracy/macro_f1 should decrease while hd_after stays near hd_before.",
            "scope": f"This correction trial uses the {args.label_panel} label panel.",
        },
    }
    with open(out_dir / "batch_adversarial_correction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(history).to_csv(out_dir / "batch_adversarial_training_history.csv", index=False)
    pd.DataFrame(summary["before_after_probe_results"]).to_csv(out_dir / "before_after_probe_metrics.csv", index=False)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adversarial batch correction for taxonomy-aware BiomeGPT")
    p.add_argument("--data_dir", default="dataset_v3")
    p.add_argument("--taxonomy_xlsx", default="dataset_v3/species_taxonomy_filled_validated_Serena.xlsx")
    p.add_argument("--checkpoint", default="dataset_v3/outputs_taxonomy_notebook/taxonomy_checkpoint_stage2.pt")
    p.add_argument(
        "--before_embeddings_npz",
        default="dataset_v3/outputs_batch_diagnostics_taxonomy/phase2_sample_prompt_embeddings.npz",
    )
    p.add_argument(
        "--batch_annotation_csv",
        default="dataset_v3/meta_pretraining_phase2_gut_batch_annotation_external_enriched.csv",
    )
    p.add_argument("--output_dir", default="dataset_v3/outputs_batch_adversarial_taxonomy")
    p.add_argument(
        "--label_panel",
        choices=["conservative_safe", "high_only", "high_medium"],
        default="conservative_safe",
    )
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--embedding_batch_size", type=int, default=16)
    p.add_argument("--min_batch_size", type=int, default=10)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument("--dab_weight", type=float, default=0.1)
    p.add_argument("--grl_lambda", type=float, default=1.0)
    p.add_argument("--dab_warmup_epochs", type=float, default=0.0)
    p.add_argument("--grl_warmup_epochs", type=float, default=0.0)
    p.add_argument("--alignment_loss", choices=["none", "centroid", "coral", "mmd"], default="none")
    p.add_argument("--alignment_weight", type=float, default=0.0)
    p.add_argument("--alignment_warmup_epochs", type=float, default=0.0)
    p.add_argument("--alignment_min_group_size", type=int, default=2)
    p.add_argument("--alignment_normalize", action="store_true")
    p.add_argument("--balanced_study_batches", action="store_true")
    p.add_argument("--studies_per_batch", type=int, default=8)
    p.add_argument("--samples_per_study", type=int, default=4)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
