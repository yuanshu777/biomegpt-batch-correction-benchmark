"""
Taxonomy-aware BiomeGPT workflow for dataset_v3.

This script extends the original reproduction scaffold with:
- rank-wise taxonomic embeddings for species tokens
- species prompts and sample prompts
- sample/species embedding extraction for UMAP/PCA visualization
- Healthy vs Diseased fine-tuning and ExVal evaluation

The defaults are intentionally close to the BiomeGPT paper, but every expensive
training setting is configurable from the command line.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]
    TensorDataset = object  # type: ignore[assignment]
    random_split = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


RANKS = ["Domain", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
UNKNOWN_TOKEN = "unknown"


@dataclass
class Metrics:
    accuracy: float
    f1: float
    auroc: float
    macro_accuracy: float
    macro_f1: float
    macro_auroc: float
    accuracy_h: float
    accuracy_d: float
    threshold: float
    n_h: int
    n_d: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_csv_or_zip(path: Path, index_col: int = 0, nrows: Optional[int] = None) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            members = [
                m
                for m in zf.namelist()
                if m.lower().endswith(".csv") and not m.endswith("/") and not m.startswith("__MACOSX/")
            ]
            if not members:
                raise RuntimeError(f"No CSV file found inside zip: {path}")
            members = sorted(members, key=lambda x: ("/" in x, x))
            with zf.open(members[0]) as f:
                return pd.read_csv(f, index_col=index_col, nrows=nrows)
    return pd.read_csv(path, index_col=index_col, nrows=nrows)


def align_abund_meta(abund: pd.DataFrame, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = abund.index.intersection(meta.index)
    if len(common) == 0:
        raise RuntimeError("No overlapping sample IDs between abundance and metadata.")
    return abund.loc[common], meta.loc[common]


def load_taxonomy(taxonomy_xlsx: Path, species: Sequence[str]) -> pd.DataFrame:
    tax = pd.read_excel(taxonomy_xlsx, sheet_name=0)
    required = ["SpeciesName"] + RANKS
    missing = [c for c in required if c not in tax.columns]
    if missing:
        raise ValueError(f"Taxonomy file is missing required columns: {missing}")
    tax = tax[required].copy()
    tax["SpeciesName"] = tax["SpeciesName"].astype(str)
    for col in RANKS:
        tax[col] = (
            tax[col]
            .astype(str)
            .str.strip()
            .replace({"": UNKNOWN_TOKEN, "nan": UNKNOWN_TOKEN, "None": UNKNOWN_TOKEN})
        )
        tax[col] = tax[col].str.replace("uknown", UNKNOWN_TOKEN, case=False, regex=False)
    tax = tax.drop_duplicates("SpeciesName").set_index("SpeciesName")

    missing_species = [s for s in species if s not in tax.index]
    if missing_species:
        raise ValueError(
            f"Taxonomy is missing {len(missing_species)} species. First missing: {missing_species[:5]}"
        )
    return tax.loc[list(species)]


def build_taxonomy_ids(tax: pd.DataFrame) -> Tuple[Dict[str, Dict[str, int]], Dict[str, np.ndarray]]:
    vocabs: Dict[str, Dict[str, int]] = {}
    ids: Dict[str, np.ndarray] = {}
    for rank in RANKS:
        values = [UNKNOWN_TOKEN] + sorted(v for v in tax[rank].astype(str).unique() if v != UNKNOWN_TOKEN)
        vocab = {value: idx for idx, value in enumerate(values)}
        vocabs[rank] = vocab
        ids[rank] = tax[rank].map(vocab).to_numpy(dtype=np.int64)
    return vocabs, ids


def rank_to_bins(nonzero_values: np.ndarray, num_bins: int) -> np.ndarray:
    n = nonzero_values.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(-nonzero_values)
    bins = np.zeros(n, dtype=np.int64)
    if n >= num_bins:
        ranks = np.arange(n, dtype=np.float64)
        assigned = num_bins - np.floor(ranks * num_bins / n).astype(np.int64)
    elif n == 1:
        assigned = np.array([num_bins], dtype=np.int64)
    else:
        ranks = np.arange(n, dtype=np.float64)
        assigned = np.round((n - 1 - ranks) * (num_bins - 1) / (n - 1)).astype(np.int64) + 1
    bins[order] = np.clip(assigned, 1, num_bins)
    return bins


def abundance_to_binned_matrix(abund: pd.DataFrame, num_bins: int) -> np.ndarray:
    arr = abund.to_numpy(dtype=np.float32)
    out = np.zeros(arr.shape, dtype=np.int64)
    for i in range(arr.shape[0]):
        nz = np.flatnonzero(arr[i] > 0)
        if nz.size:
            out[i, nz] = rank_to_bins(arr[i, nz], num_bins)
    return out


def make_hd_labels(meta: pd.DataFrame, phenotype_col: str = "Phenotype_fullname") -> np.ndarray:
    if phenotype_col not in meta.columns:
        phenotype_col = "Phenotype"
    phenotype = meta[phenotype_col].astype(str).str.lower()
    return (phenotype != "healthy").astype(np.int64).to_numpy()


def stratified_cap(
    abund: pd.DataFrame,
    meta: pd.DataFrame,
    labels: np.ndarray,
    max_samples: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if max_samples <= 0 or max_samples >= len(labels):
        return abund, meta, labels
    rng = np.random.default_rng(seed)
    unique = sorted(np.unique(labels))
    chosen_parts = []
    for label in unique:
        idx = np.flatnonzero(labels == label)
        take = min(len(idx), max(1, max_samples // len(unique)))
        chosen_parts.append(rng.choice(idx, size=take, replace=False))
    chosen = np.concatenate(chosen_parts)
    if len(chosen) < max_samples:
        remaining = np.setdiff1d(np.arange(len(labels)), chosen)
        extra = rng.choice(remaining, size=min(max_samples - len(chosen), len(remaining)), replace=False)
        chosen = np.concatenate([chosen, extra])
    rng.shuffle(chosen)
    capped_abund = abund.iloc[chosen]
    capped_meta = meta.loc[capped_abund.index]
    return capped_abund, capped_meta, labels[chosen]


def stratified_train_val_indices(
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a stratified train/validation split for threshold and epoch selection."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_parts = []
    val_parts = []
    for label in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n_val = max(1, int(round(val_fraction * len(idx))))
        n_val = min(n_val, max(len(idx) - 1, 1))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def add_synthetic_minority(
    abund: pd.DataFrame,
    labels: np.ndarray,
    minority_label: int = 1,
    std: float = 5.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = {int(v): int((labels == v).sum()) for v in np.unique(labels)}
    majority = max(counts.values())
    current = counts.get(minority_label, 0)
    needed = majority - current
    if needed <= 0:
        return abund, labels

    minority_idx = np.flatnonzero(labels == minority_label)
    chosen = rng.choice(minority_idx, size=needed, replace=True)
    base = abund.to_numpy(dtype=np.float32)[chosen].copy()
    nonzero = base > 0
    noise = rng.normal(loc=0.0, scale=std, size=base.shape).astype(np.float32)
    synthetic = base.copy()
    synthetic[nonzero] = synthetic[nonzero] + noise[nonzero]
    synthetic[~nonzero] = 0.0
    synthetic = np.clip(synthetic, 0.0, None)

    synthetic_index = [f"synthetic_D_{i:06d}" for i in range(needed)]
    synthetic_df = pd.DataFrame(synthetic, index=synthetic_index, columns=abund.columns)
    out_abund = pd.concat([abund, synthetic_df], axis=0)
    out_labels = np.concatenate([labels, np.full(needed, minority_label, dtype=np.int64)])
    return out_abund, out_labels


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for group_idx, count in enumerate(counts):
        if count > 1:
            ties = inv == group_idx
            ranks[ties] = ranks[ties].mean()
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_metrics(y_true: np.ndarray, prob_d: np.ndarray, threshold: float) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    prob_d = np.asarray(prob_d).astype(float)
    pred = (prob_d >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision_d = tp / max(tp + fp, 1)
    recall_d = tp / max(tp + fn, 1)
    f1_d = 2 * precision_d * recall_d / max(precision_d + recall_d, 1e-12)

    precision_h = tn / max(tn + fn, 1)
    recall_h = tn / max(tn + fp, 1)
    f1_h = 2 * precision_h * recall_h / max(precision_h + recall_h, 1e-12)

    acc_h = recall_h
    acc_d = recall_d
    auc_d = binary_auc(y_true, prob_d)
    auc_h = binary_auc(1 - y_true, 1 - prob_d)

    return Metrics(
        accuracy=float(accuracy),
        f1=float(f1_d),
        auroc=float(auc_d),
        macro_accuracy=float((acc_h + acc_d) / 2),
        macro_f1=float((f1_h + f1_d) / 2),
        macro_auroc=float(np.nanmean([auc_h, auc_d])),
        accuracy_h=float(acc_h),
        accuracy_d=float(acc_d),
        threshold=float(threshold),
        n_h=int((y_true == 0).sum()),
        n_d=int((y_true == 1).sum()),
    )


def confusion_matrix_table(y_true: np.ndarray, prob_d: np.ndarray, threshold: float) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob_d).astype(float) >= threshold).astype(int)
    return pd.DataFrame(
        [
            {
                "true_label": "Healthy",
                "pred_healthy": int(((y_true == 0) & (pred == 0)).sum()),
                "pred_diseased": int(((y_true == 0) & (pred == 1)).sum()),
            },
            {
                "true_label": "Diseased",
                "pred_healthy": int(((y_true == 1) & (pred == 0)).sum()),
                "pred_diseased": int(((y_true == 1) & (pred == 1)).sum()),
            },
        ]
    )


def metric_sanity_checks(y_true: np.ndarray, prob_d: np.ndarray, metrics: Metrics) -> Dict[str, object]:
    """Independent checks for formulas that are easy to misreport."""
    cm = confusion_matrix_table(y_true, prob_d, metrics.threshold)
    tn = int(cm.loc[cm["true_label"] == "Healthy", "pred_healthy"].iloc[0])
    fp = int(cm.loc[cm["true_label"] == "Healthy", "pred_diseased"].iloc[0])
    fn = int(cm.loc[cm["true_label"] == "Diseased", "pred_healthy"].iloc[0])
    tp = int(cm.loc[cm["true_label"] == "Diseased", "pred_diseased"].iloc[0])
    n_h = max(tn + fp, 1)
    n_d = max(tp + fn, 1)
    acc_h = tn / n_h
    acc_d = tp / n_d
    pred = (np.asarray(prob_d) >= metrics.threshold).astype(int)
    return {
        "threshold_used": metrics.threshold,
        "probability_range": {
            "min": float(np.min(prob_d)),
            "median": float(np.median(prob_d)),
            "max": float(np.max(prob_d)),
        },
        "prediction_counts": {
            "pred_healthy": int((pred == 0).sum()),
            "pred_diseased": int((pred == 1).sum()),
        },
        "class_accuracy_formula_check": {
            "accuracy_h_from_confusion_matrix": float(acc_h),
            "accuracy_d_from_confusion_matrix": float(acc_d),
            "matches_metrics": bool(
                np.isclose(acc_h, metrics.accuracy_h) and np.isclose(acc_d, metrics.accuracy_d)
            ),
        },
        "auroc_uses_probabilities": bool(len(np.unique(np.asarray(prob_d))) > 2),
        "flags": {
            "all_predictions_one_class": bool(len(np.unique(pred)) == 1),
            "low_probability_spread": bool(float(np.std(prob_d)) < 0.01),
        },
    }


def save_probability_histogram(
    y_true: np.ndarray,
    prob_d: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    prob_d = np.asarray(prob_d).astype(float)
    plt.figure(figsize=(7, 4.5))
    plt.hist(prob_d[y_true == 0], bins=20, alpha=0.65, label="Healthy", color="#4C78A8")
    plt.hist(prob_d[y_true == 1], bins=20, alpha=0.65, label="Diseased", color="#F58518")
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1.4, label=f"threshold={threshold:.1f}")
    plt.xlabel("Predicted probability of Diseased")
    plt.ylabel("Sample count")
    plt.title("ExVal prediction probability distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def best_threshold(y_true: np.ndarray, prob_d: np.ndarray) -> Metrics:
    candidates = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]
    metrics = [compute_metrics(y_true, prob_d, t) for t in candidates]
    return max(metrics, key=lambda m: (m.macro_f1, m.macro_accuracy))


def knn_label_purity(
    embeddings: np.ndarray,
    labels: Sequence[str],
    k_values: Sequence[int] = (5, 10, 20),
) -> Dict[str, float]:
    """Report how often nearest neighbors share the same taxonomy label."""
    x = np.asarray(embeddings, dtype=np.float32)
    labels_arr = np.asarray([str(v) for v in labels])
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    out: Dict[str, float] = {}
    for k in k_values:
        kk = min(k, max(1, len(labels_arr) - 1))
        nn = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
        purity = (labels_arr[nn] == labels_arr[:, None]).mean(axis=1)
        out[f"knn_purity_k{k}"] = float(np.mean(purity))
    return out


def binary_centroid_summary(embeddings: np.ndarray, labels: Sequence[str]) -> Dict[str, object]:
    """Small numeric companion to UMAP plots for two-class sample embeddings."""
    x = np.asarray(embeddings, dtype=np.float32)
    label_series = pd.Series(labels).astype(str)
    groups = sorted(label_series.unique())
    summary: Dict[str, object] = {"counts": label_series.value_counts().to_dict()}
    if len(groups) == 2:
        a = x[label_series.to_numpy() == groups[0]]
        b = x[label_series.to_numpy() == groups[1]]
        ca = a.mean(axis=0)
        cb = b.mean(axis=0)
        pooled = float((a.std(axis=0).mean() + b.std(axis=0).mean()) / 2)
        summary["groups"] = groups
        summary["centroid_distance"] = float(np.linalg.norm(ca - cb))
        summary["centroid_distance_over_pooled_std"] = float(summary["centroid_distance"] / max(pooled, 1e-8))
    return summary


if TORCH_AVAILABLE:

    class SpeciesBinDataset(Dataset):
        def __init__(self, binned_matrix: np.ndarray):
            self.x = torch.from_numpy(binned_matrix).long()

        def __len__(self) -> int:
            return self.x.shape[0]

        def __getitem__(self, idx: int) -> torch.Tensor:
            return self.x[idx]


    class TaxonomyBiomeGPT(nn.Module):
        def __init__(
            self,
            taxonomy_vocab_sizes: Dict[str, int],
            taxonomy_ids: Dict[str, np.ndarray],
            num_bins: int = 32,
            d_model: int = 512,
            nhead: int = 8,
            num_layers: int = 8,
            ff_dim: int = 512,
            dropout: float = 0.1,
            num_classes: int = 2,
        ):
            super().__init__()
            self.num_species = len(next(iter(taxonomy_ids.values())))
            self.num_bins = num_bins
            self.nhead = nhead
            self.d_model = d_model

            self.rank_embeddings = nn.ModuleDict(
                {rank: nn.Embedding(taxonomy_vocab_sizes[rank], d_model) for rank in RANKS}
            )
            for rank in RANKS:
                self.register_buffer(
                    f"{rank.lower()}_ids",
                    torch.from_numpy(taxonomy_ids[rank].copy()).long().unsqueeze(0),
                    persistent=True,
                )
            self.cls_species_emb = nn.Parameter(torch.empty(d_model))
            nn.init.normal_(self.cls_species_emb, mean=0.0, std=0.02)

            self.abund_mlp = nn.Sequential(
                nn.Linear(1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.ln_species = nn.LayerNorm(d_model)
            self.ln_abund = nn.LayerNorm(d_model)

            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                activation="relu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.reconstruction_head = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        def species_prompt(self, species_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Return learned species embeddings, analogous to scGPT gene embeddings."""
            rank_sum = None
            for rank in RANKS:
                ids = getattr(self, f"{rank.lower()}_ids")
                emb = self.rank_embeddings[rank](ids)
                rank_sum = emb if rank_sum is None else rank_sum + emb
            species_emb = self.ln_species(rank_sum.squeeze(0))
            if species_indices is not None:
                species_emb = species_emb[species_indices]
            return species_emb

        def _tokenize(self, bins: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            bsz, _seq_len = bins.shape
            input_bins = bins.clone()
            if mask is not None:
                input_bins[mask] = 0

            species_tok = self.species_prompt().unsqueeze(0).expand(bsz, -1, -1)
            cls_tok = self.ln_species(self.cls_species_emb).view(1, 1, -1).expand(bsz, -1, -1)
            species_tok = torch.cat([cls_tok, species_tok], dim=1)

            cls_bin = torch.zeros((bsz, 1), dtype=torch.long, device=bins.device)
            all_bins = torch.cat([cls_bin, input_bins], dim=1)
            abund_tok = self.ln_abund(self.abund_mlp((all_bins.float() / max(self.num_bins, 1)).unsqueeze(-1)))
            return species_tok + abund_tok

        def _build_pretrain_attention_mask(self, bins: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            bsz, seq_len = bins.shape
            length = seq_len + 1
            device = bins.device
            nonzero = bins > 0
            unmasked = nonzero & (~mask)
            masked = nonzero & mask

            allow = torch.zeros((bsz, length, length), dtype=torch.bool, device=device)
            allow[:, 0, 0] = True
            allow[:, 0, 1:] = unmasked
            allow[:, 1:, 0] = nonzero

            unmasked_q = unmasked.unsqueeze(2)
            masked_q = masked.unsqueeze(2)
            unmasked_k = unmasked.unsqueeze(1)
            species_allow = (unmasked_q & unmasked_k) | (masked_q & unmasked_k)
            eye = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
            species_allow = species_allow | eye
            allow[:, 1:, 1:] = species_allow

            disallow = ~allow
            return disallow.unsqueeze(1).expand(bsz, self.nhead, length, length).reshape(
                bsz * self.nhead, length, length
            )

        def _build_inference_attention_mask(self, bins: torch.Tensor) -> torch.Tensor:
            bsz, seq_len = bins.shape
            length = seq_len + 1
            device = bins.device
            nonzero = bins > 0
            allow = torch.zeros((bsz, length, length), dtype=torch.bool, device=device)
            allow[:, 0, 0] = True
            allow[:, 0, 1:] = nonzero
            allow[:, 1:, 0] = nonzero
            species_allow = nonzero.unsqueeze(2) & nonzero.unsqueeze(1)
            eye = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
            allow[:, 1:, 1:] = species_allow | eye
            disallow = ~allow
            return disallow.unsqueeze(1).expand(bsz, self.nhead, length, length).reshape(
                bsz * self.nhead, length, length
            )

        def encode(self, bins: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            x = self._tokenize(bins, mask=mask)
            attn_mask = (
                self._build_pretrain_attention_mask(bins, mask)
                if mask is not None
                else self._build_inference_attention_mask(bins)
            )
            return self.encoder(x, mask=attn_mask)

        def forward_pretrain(self, bins: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            h = self.encode(bins, mask=mask)
            return self.reconstruction_head(h[:, 1:, :]).squeeze(-1)

        def sample_prompt(self, bins: torch.Tensor) -> torch.Tensor:
            """Return <cls> sample embeddings, analogous to scGPT cell embeddings."""
            return self.encode(bins, mask=None)[:, 0, :]

        def forward_classify(self, bins: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.sample_prompt(bins))


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. Install dependencies from requirements_taxonomy_pipeline.txt.")


def build_model_from_species(
    species: Sequence[str],
    taxonomy_xlsx: Path,
    num_bins: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    ff_dim: int,
    dropout: float,
) -> Tuple["TaxonomyBiomeGPT", pd.DataFrame, Dict[str, Dict[str, int]]]:
    require_torch()
    tax = load_taxonomy(taxonomy_xlsx, species)
    vocabs, ids = build_taxonomy_ids(tax)
    model = TaxonomyBiomeGPT(
        taxonomy_vocab_sizes={rank: len(vocabs[rank]) for rank in RANKS},
        taxonomy_ids=ids,
        num_bins=num_bins,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    )
    return model, tax, vocabs


def build_mask(batch_bins: torch.Tensor, mask_ratio: float, rng: torch.Generator) -> torch.Tensor:
    bsz, seq_len = batch_bins.shape
    mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=batch_bins.device)
    for i in range(bsz):
        nz = torch.nonzero(batch_bins[i] > 0, as_tuple=False).squeeze(-1)
        if nz.numel() == 0:
            continue
        k = max(1, int(round(mask_ratio * nz.numel())))
        perm = torch.randperm(nz.numel(), generator=rng, device=batch_bins.device)
        mask[i, nz[perm[:k]]] = True
    return mask


def pretrain_epoch(
    model: "TaxonomyBiomeGPT",
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float,
    rng: torch.Generator,
    mixed_precision: bool,
) -> float:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(mixed_precision and device.type == "cuda"))
    losses: List[float] = []
    for batch in loader:
        bins = batch.to(device, non_blocking=True)
        mask = build_mask(bins, mask_ratio, rng)
        valid = mask & (bins > 0)
        if valid.sum().item() == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            pred = model.forward_pretrain(bins, mask)
            loss = F.mse_loss(pred[valid], bins[valid].float())
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
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def save_checkpoint(
    path: Path,
    model: "TaxonomyBiomeGPT",
    optimizer: Optional[torch.optim.Optimizer],
    species: Sequence[str],
    taxonomy: pd.DataFrame,
    args: argparse.Namespace,
    extra: Optional[dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "species": list(species),
        "taxonomy": taxonomy.reset_index().to_dict(orient="list"),
        "args": vars(args),
        "extra": extra or {},
    }
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint_model(path: Path, taxonomy_xlsx: Path, device: torch.device) -> Tuple["TaxonomyBiomeGPT", List[str], dict]:
    require_torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_args = payload.get("args", {})
    species = list(payload["species"])
    model, _tax, _vocabs = build_model_from_species(
        species=species,
        taxonomy_xlsx=taxonomy_xlsx,
        num_bins=int(saved_args.get("bins", saved_args.get("num_bins", 32))),
        d_model=int(saved_args.get("d_model", 512)),
        nhead=int(saved_args.get("nhead", 8)),
        num_layers=int(saved_args.get("num_layers", 8)),
        ff_dim=int(saved_args.get("ff_dim", 512)),
        dropout=float(saved_args.get("dropout", 0.1)),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device)
    return model, species, payload


def run_pretrain(args: argparse.Namespace) -> None:
    require_torch()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    device = torch.device(args.device)

    phase1_abund = load_csv_or_zip(data_dir / "abund_pretraining_phase1_gut_and_nongut.csv.zip")
    phase1_meta = load_csv_or_zip(data_dir / "meta_pretraining_phase1_gut_and_nongut.csv")
    phase1_abund, phase1_meta = align_abund_meta(phase1_abund, phase1_meta)
    if args.max_phase1_samples:
        phase1_abund = phase1_abund.head(args.max_phase1_samples)
        phase1_meta = phase1_meta.loc[phase1_abund.index]
    species = phase1_abund.columns.tolist()
    model, tax, _vocabs = build_model_from_species(
        species, Path(args.taxonomy_xlsx), args.bins, args.d_model, args.nhead, args.num_layers, args.ff_dim, args.dropout
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator(device=device.type if device.type in {"cpu", "cuda"} else "cpu").manual_seed(args.seed)

    phase1_bins = abundance_to_binned_matrix(phase1_abund, args.bins)
    phase1_loader = DataLoader(
        SpeciesBinDataset(phase1_bins),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    history = []
    for epoch in range(1, args.epochs_phase1 + 1):
        loss = pretrain_epoch(model, phase1_loader, optimizer, device, args.mask_ratio, rng, args.mixed_precision)
        history.append({"stage": "phase1", "epoch": epoch, "loss": loss})
        print(f"[phase1] epoch {epoch:03d}/{args.epochs_phase1} loss={loss:.6f}")
    save_checkpoint(out_dir / "taxonomy_checkpoint_stage1.pt", model, optimizer, species, tax, args, {"history": history})

    phase2_abund = load_csv_or_zip(data_dir / "abund_pretraining_phase2_gut.csv.zip")
    phase2_meta = load_csv_or_zip(data_dir / "meta_pretraining_phase2_gut.csv")
    phase2_abund, phase2_meta = align_abund_meta(phase2_abund, phase2_meta)
    if args.max_phase2_samples:
        phase2_abund = phase2_abund.head(args.max_phase2_samples)
        phase2_meta = phase2_meta.loc[phase2_abund.index]
    phase2_abund = phase2_abund.reindex(columns=species, fill_value=0.0)
    phase2_bins = abundance_to_binned_matrix(phase2_abund, args.bins)
    phase2_loader = DataLoader(
        SpeciesBinDataset(phase2_bins),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    for epoch in range(1, args.epochs_phase2 + 1):
        loss = pretrain_epoch(model, phase2_loader, optimizer, device, args.mask_ratio, rng, args.mixed_precision)
        history.append({"stage": "phase2", "epoch": epoch, "loss": loss})
        print(f"[phase2] epoch {epoch:03d}/{args.epochs_phase2} loss={loss:.6f}")
    save_checkpoint(out_dir / "taxonomy_checkpoint_stage2.pt", model, optimizer, species, tax, args, {"history": history})


def project_2d(x: np.ndarray, seed: int) -> Tuple[np.ndarray, str]:
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_neighbors=20, min_dist=0.15, metric="cosine", random_state=seed)
        return reducer.fit_transform(x), "UMAP"
    except ModuleNotFoundError:
        try:
            from sklearn.decomposition import PCA  # type: ignore

            return PCA(n_components=2, random_state=seed).fit_transform(x), "PCA"
        except ModuleNotFoundError:
            x = x - x.mean(axis=0, keepdims=True)
            u, s, _vt = np.linalg.svd(x, full_matrices=False)
            return u[:, :2] * s[:2], "PCA_SVD"


def save_scatter(
    coords: np.ndarray,
    labels: Sequence[str],
    title: str,
    path: Path,
    max_legend: int = 20,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = pd.Series(labels).astype(str)
    top = labels.value_counts().head(max_legend).index
    plot_labels = labels.where(labels.isin(top), other="Other")
    groups = sorted(plot_labels.unique())
    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(8, 6))
    for i, group in enumerate(groups):
        idx = plot_labels == group
        plt.scatter(coords[idx, 0], coords[idx, 1], s=8, alpha=0.75, color=cmap(i % 20), label=group)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(markerscale=2, fontsize=7, frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def extract_sample_embeddings(
    model: "TaxonomyBiomeGPT",
    bins: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(SpeciesBinDataset(bins), batch_size=batch_size, shuffle=False)
    embs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embs.append(model.sample_prompt(batch).detach().cpu().numpy())
    return np.vstack(embs)


def run_embeddings(args: argparse.Namespace) -> None:
    require_torch()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    model, species, _payload = load_checkpoint_model(Path(args.checkpoint), Path(args.taxonomy_xlsx), device)
    tax = load_taxonomy(Path(args.taxonomy_xlsx), species)

    if args.sample_umap:
        abund = load_csv_or_zip(data_dir / "abund_pretraining_phase1_gut_and_nongut.csv.zip")
        meta = load_csv_or_zip(data_dir / "meta_pretraining_phase1_gut_and_nongut.csv")
        abund, meta = align_abund_meta(abund, meta)
        if args.max_samples:
            labels = (meta["body_site"].astype(str) != "stool").astype(np.int64).to_numpy()
            abund, meta, _labels = stratified_cap(abund, meta, labels, args.max_samples, args.seed)
        abund = abund.reindex(columns=species, fill_value=0.0)
        bins = abundance_to_binned_matrix(abund, args.bins)
        emb = extract_sample_embeddings(model, bins, device, args.batch_size)
        coords, method = project_2d(emb, args.seed)
        body_site = meta["body_site"].astype(str).where(meta["body_site"].astype(str) == "stool", "non-gut")
        with open(out_dir / "sample_prompt_gut_vs_nongut_summary.json", "w", encoding="utf-8") as f:
            json.dump(binary_centroid_summary(emb, body_site), f, indent=2)
        sample_df = pd.DataFrame(
            {"sample_id": abund.index, "x": coords[:, 0], "y": coords[:, 1], "body_site": body_site.to_numpy()}
        )
        sample_df.to_csv(out_dir / "sample_prompt_gut_vs_nongut_embeddings.csv", index=False)
        save_scatter(coords, body_site, f"{method}: sample prompts, gut vs non-gut", out_dir / "sample_prompt_gut_vs_nongut.png")

    if args.species_umap:
        model.eval()
        with torch.no_grad():
            emb = model.species_prompt().detach().cpu().numpy()
        coords, method = project_2d(emb, args.seed)
        species_df = tax.reset_index().rename(columns={"index": "SpeciesName"})
        species_df.insert(1, "x", coords[:, 0])
        species_df.insert(2, "y", coords[:, 1])
        species_df.to_csv(out_dir / "species_prompt_taxonomy_embeddings.csv", index=False)
        purity_summary = {
            rank: knn_label_purity(emb, tax[rank].astype(str).tolist())
            for rank in ["Genus", "Family", "Order", "Phylum"]
        }
        with open(out_dir / "species_prompt_taxonomy_neighbor_purity.json", "w", encoding="utf-8") as f:
            json.dump(purity_summary, f, indent=2)
        for rank in ["Genus", "Family", "Order", "Phylum"]:
            save_scatter(
                coords,
                tax[rank].astype(str).tolist(),
                f"{method}: species prompts colored by {rank}",
                out_dir / f"species_prompt_by_{rank.lower()}.png",
            )


def maybe_lasso_feature_mask(
    train_abund: pd.DataFrame,
    labels: np.ndarray,
    seed: int,
    enabled: bool,
) -> Optional[np.ndarray]:
    if not enabled:
        return None
    try:
        from sklearn.linear_model import LogisticRegressionCV  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.pipeline import make_pipeline  # type: ignore
    except ModuleNotFoundError:
        print("scikit-learn is not installed; skipping L1/Lasso-style feature selection.")
        return None

    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="l1",
            solver="saga",
            scoring="f1_macro",
            max_iter=3000,
            n_jobs=-1,
            random_state=seed,
        ),
    )
    model.fit(train_abund.to_numpy(dtype=np.float32), labels)
    clf = model.named_steps["logisticregressioncv"]
    coef = np.abs(clf.coef_).reshape(-1)
    mask = coef > 1e-8
    if mask.sum() == 0:
        print("L1 feature selection selected zero features; ignoring mask.")
        return None
    print(f"L1 feature selection retained {int(mask.sum())}/{len(mask)} species.")
    return mask


def write_data_contract_artifacts(
    out_dir: Path,
    data_dir: Path,
    exval_dir: Path,
    taxonomy_xlsx: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    phase1_cols = load_csv_or_zip(data_dir / "abund_pretraining_phase1_gut_and_nongut.csv.zip", nrows=0).columns.tolist()
    phase2_cols = load_csv_or_zip(data_dir / "abund_pretraining_phase2_gut.csv.zip", nrows=0).columns.tolist()
    prev3_cols = load_csv_or_zip(data_dir / "abund_finetuning_gut_prev3.csv.zip", nrows=0).columns.tolist()
    exval_cols = load_csv_or_zip(exval_dir / "df_validation_data.csv", nrows=0).columns.tolist()
    tax = pd.read_excel(taxonomy_xlsx, sheet_name=0)
    tax_species = set(tax["SpeciesName"].astype(str))

    alignment_rows = [
        {
            "comparison": "phase1_species_in_taxonomy",
            "left_count": len(phase1_cols),
            "overlap_count": len(set(phase1_cols) & tax_species),
            "missing_count": len(set(phase1_cols) - tax_species),
            "missing_examples": ";".join(sorted(set(phase1_cols) - tax_species)[:10]),
        },
        {
            "comparison": "prev3_species_in_exval",
            "left_count": len(prev3_cols),
            "overlap_count": len(set(prev3_cols) & set(exval_cols)),
            "missing_count": len(set(prev3_cols) - set(exval_cols)),
            "missing_examples": ";".join(sorted(set(prev3_cols) - set(exval_cols))[:10]),
        },
        {
            "comparison": "phase2_species_in_phase1",
            "left_count": len(phase2_cols),
            "overlap_count": len(set(phase2_cols) & set(phase1_cols)),
            "missing_count": len(set(phase2_cols) - set(phase1_cols)),
            "missing_examples": ";".join(sorted(set(phase2_cols) - set(phase1_cols))[:10]),
        },
    ]
    pd.DataFrame(alignment_rows).to_csv(out_dir / "species_alignment_summary.csv", index=False)

    rank_rows = []
    for rank in RANKS:
        raw = tax[rank].astype(str).str.strip().str.lower()
        rank_rows.append(
            {
                "rank": rank,
                "unique_values": int(tax[rank].astype(str).nunique()),
                "unknown_like_values": int(raw.isin([UNKNOWN_TOKEN, "uknown", "nan", "none", ""]).sum()),
            }
        )
    pd.DataFrame(rank_rows).to_csv(out_dir / "taxonomy_completeness_summary.csv", index=False)

    train_meta = load_csv_or_zip(data_dir / "meta_finetuning_gut_prev3.csv")
    ex_meta = load_csv_or_zip(exval_dir / "df_validation_data_metadata.csv")
    y_train = make_hd_labels(train_meta)
    y_ex = make_hd_labels(ex_meta, phenotype_col="Phenotype")
    pd.DataFrame(
        [
            {"dataset": "train_prev3", "healthy": int((y_train == 0).sum()), "diseased": int((y_train == 1).sum())},
            {"dataset": "exval", "healthy": int((y_ex == 0).sum()), "diseased": int((y_ex == 1).sum())},
        ]
    ).to_csv(out_dir / "label_balance_summary.csv", index=False)

    summary = {
        "phase1_species_count": len(phase1_cols),
        "phase2_species_count": len(phase2_cols),
        "prev3_species_count": len(prev3_cols),
        "exval_species_count": len(exval_cols),
        "taxonomy_species_count": int(tax["SpeciesName"].astype(str).nunique()),
        "prev3_species_missing_from_exval": sorted(set(prev3_cols) - set(exval_cols)),
    }
    with open(out_dir / "data_contract_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def train_classifier_epoch(
    model: "TaxonomyBiomeGPT",
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor,
    mixed_precision: bool,
) -> float:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(mixed_precision and device.type == "cuda"))
    losses = []
    for bins, labels in loader:
        bins = bins.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = model.forward_classify(bins)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
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
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def predict_classifier(
    model: "TaxonomyBiomeGPT",
    bins: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(SpeciesBinDataset(bins), batch_size=batch_size, shuffle=False)
    probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model.forward_classify(batch)
            probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probs)


def prepare_finetune_data(args: argparse.Namespace, species: Sequence[str]) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    data_dir = Path(args.data_dir)
    exval_dir = Path(args.exval_dir)
    train_abund = load_csv_or_zip(data_dir / "abund_finetuning_gut_prev3.csv.zip")
    train_meta = load_csv_or_zip(data_dir / "meta_finetuning_gut_prev3.csv")
    train_abund, train_meta = align_abund_meta(train_abund, train_meta)
    y_train = make_hd_labels(train_meta)
    if args.max_train_samples:
        train_abund, train_meta, y_train = stratified_cap(
            train_abund, train_meta, y_train, args.max_train_samples, args.seed
        )
    train_abund = train_abund.reindex(columns=species, fill_value=0.0)

    ex_abund = load_csv_or_zip(exval_dir / "df_validation_data.csv")
    ex_meta = load_csv_or_zip(exval_dir / "df_validation_data_metadata.csv")
    ex_abund, ex_meta = align_abund_meta(ex_abund, ex_meta)
    y_ex = make_hd_labels(ex_meta, phenotype_col="Phenotype")
    if args.max_exval_samples:
        ex_abund, ex_meta, y_ex = stratified_cap(ex_abund, ex_meta, y_ex, args.max_exval_samples, args.seed)
    ex_abund = ex_abund.reindex(columns=species, fill_value=0.0)
    return train_abund, y_train, ex_abund, y_ex


def run_finetune(args: argparse.Namespace) -> None:
    require_torch()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, checkpoint_species, _payload = load_checkpoint_model(Path(args.checkpoint), Path(args.taxonomy_xlsx), device)

    train_abund, y_train, ex_abund, y_ex = prepare_finetune_data(args, checkpoint_species)
    train_idx, val_idx = stratified_train_val_indices(y_train, args.selection_fraction, args.seed)

    # Feature selection is fit only on the selection-training split so threshold/epoch
    # selection does not peek at validation labels.
    feature_mask = maybe_lasso_feature_mask(
        train_abund.iloc[train_idx],
        y_train[train_idx],
        args.seed,
        args.use_l1_feature_selection,
    )
    if feature_mask is not None:
        train_abund = train_abund.copy()
        ex_abund = ex_abund.copy()
        train_abund.loc[:, ~feature_mask] = 0.0
        ex_abund.loc[:, ~feature_mask] = 0.0

    # Internal split is used only to choose threshold/epoch. Final requested metrics are on ExVal.
    selection_train_abund = train_abund.iloc[train_idx]
    selection_train_labels = y_train[train_idx]
    if args.augment_diseased:
        selection_train_abund, selection_train_labels = add_synthetic_minority(
            selection_train_abund,
            selection_train_labels,
            minority_label=1,
            std=args.synthetic_std,
            seed=args.seed,
        )
    selection_train_bins = abundance_to_binned_matrix(selection_train_abund, args.bins)
    val_abund = train_abund.iloc[val_idx]
    val_bins = abundance_to_binned_matrix(val_abund, args.bins)
    val_labels = y_train[val_idx]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(selection_train_bins).long(), torch.from_numpy(selection_train_labels).long()),
        batch_size=args.batch_size,
        shuffle=True,
    )

    counts = np.bincount(selection_train_labels, minlength=2).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"macro_f1": -1.0, "epoch": 0, "threshold": 0.5, "state": None}
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_classifier_epoch(model, train_loader, optimizer, device, class_weights, args.mixed_precision)
        val_prob = predict_classifier(model, val_bins, device, args.batch_size)
        val_metrics = best_threshold(val_labels, val_prob)
        history.append({"epoch": epoch, "loss": loss, **asdict(val_metrics)})
        print(
            f"[selection] epoch {epoch:03d}/{args.epochs} loss={loss:.6f} "
            f"macro_f1={val_metrics.macro_f1:.4f} threshold={val_metrics.threshold:.1f}"
        )
        if val_metrics.macro_f1 > best["macro_f1"]:
            best = {
                "macro_f1": val_metrics.macro_f1,
                "epoch": epoch,
                "threshold": val_metrics.threshold,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    # Reload pretrained checkpoint and retrain on the entire training dataset for the selected epoch count.
    model, checkpoint_species, _payload = load_checkpoint_model(Path(args.checkpoint), Path(args.taxonomy_xlsx), device)
    full_abund = train_abund
    full_labels = y_train
    if args.augment_diseased:
        full_abund, full_labels = add_synthetic_minority(
            full_abund, full_labels, minority_label=1, std=args.synthetic_std, seed=args.seed
        )
    full_bins = abundance_to_binned_matrix(full_abund, args.bins)
    full_loader = DataLoader(
        TensorDataset(torch.from_numpy(full_bins).long(), torch.from_numpy(full_labels).long()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    full_counts = np.bincount(full_labels, minlength=2).astype(np.float32)
    full_weights = torch.tensor(full_counts.sum() / np.maximum(full_counts, 1.0), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    selected_epochs = int(best["epoch"] or args.epochs)
    for epoch in range(1, selected_epochs + 1):
        loss = train_classifier_epoch(model, full_loader, optimizer, device, full_weights, args.mixed_precision)
        print(f"[final] epoch {epoch:03d}/{selected_epochs} loss={loss:.6f}")

    ex_bins = abundance_to_binned_matrix(ex_abund, args.bins)
    ex_prob = predict_classifier(model, ex_bins, device, args.batch_size)
    ex_metrics = compute_metrics(y_ex, ex_prob, float(best["threshold"]))
    pred = (ex_prob >= ex_metrics.threshold).astype(int)
    confusion_df = confusion_matrix_table(y_ex, ex_prob, ex_metrics.threshold)
    confusion_df.to_csv(out_dir / "exval_confusion_matrix.csv", index=False)
    with open(out_dir / "exval_metric_sanity_checks.json", "w", encoding="utf-8") as f:
        json.dump(metric_sanity_checks(y_ex, ex_prob, ex_metrics), f, indent=2)
    save_probability_histogram(y_ex, ex_prob, ex_metrics.threshold, out_dir / "exval_probability_histogram.png")
    pred_df = pd.DataFrame(
        {
            "sample_id": ex_abund.index,
            "true_label": y_ex,
            "prob_diseased": ex_prob,
            "pred_label": pred,
            "threshold": ex_metrics.threshold,
        }
    )
    pred_df.to_csv(out_dir / "exval_hd_predictions.csv", index=False)
    with open(out_dir / "exval_hd_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_epoch": selected_epochs,
                "selection_history": history,
                "exval_metrics": asdict(ex_metrics),
                "confusion_matrix": confusion_df.to_dict(orient="records"),
                "feature_selection_retained": int(feature_mask.sum()) if feature_mask is not None else None,
                "augmented_training_size": int(len(full_labels)),
            },
            f,
            indent=2,
        )
    save_checkpoint(
        out_dir / "taxonomy_hd_finetuned.pt",
        model,
        optimizer,
        checkpoint_species,
        load_taxonomy(Path(args.taxonomy_xlsx), checkpoint_species),
        args,
        {"exval_metrics": asdict(ex_metrics), "selected_epoch": selected_epochs},
    )
    print(json.dumps(asdict(ex_metrics), indent=2))


def write_report(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = """# Taxonomy-Aware BiomeGPT Reproduction Plan and Report

## Research Goal

This workflow asks what can be learned from the pretrained BiomeGPT species model before and after supervised fine-tuning. The key idea is that the model should learn two interpretable representation spaces: sample embeddings, analogous to cell embeddings in scGPT, and species embeddings, analogous to gene embeddings in scGPT.

## Data Inventory

The phase-1 pretraining matrix contains 1,012 species, and all 1,012 have taxonomy entries in `species_taxonomy_filled_validated_Serena.xlsx`. The `_prev3` fine-tuning matrix contains 513 gut-filtered species. The external validation matrix contains 927 samples and overlaps with 512 of those 513 `_prev3` species, so the evaluation can be performed with only one missing training species set to zero in ExVal.

## Taxonomic Hierarchy Implementation

The original species token is replaced by a sum of rank-specific embeddings:

`Domain + Kingdom + Phylum + Class + Order + Family + Genus + Species`.

This makes taxonomy part of the inductive bias. Species from the same genus share the same genus embedding; genera from the same family share the same family embedding; and so on. Biomedical interpretation becomes easier because clusters in species-embedding space can be read as learned taxonomic organization rather than arbitrary token proximity.

## Species Prompt

The species prompt returns the learned species representation from the taxonomy-composed embedding table. It is analogous to the gene embedding in scGPT. UMAPs colored by genus, family, or order test whether the model organizes microbial taxa in a biologically coherent way. If same-genus or same-family organisms cluster, the model has internalized taxonomic structure that can support biomarker interpretation.

## Sample Prompt

The sample prompt returns the final `<cls>` embedding for a microbiome sample. It is analogous to the cell embedding in scGPT. A gut vs non-gut UMAP tests whether unsupervised pretraining learns body-site structure without explicit labels. Clear separation suggests that BiomeGPT captures community-level ecological signatures, not just individual high-abundance taxa.

## Healthy vs Diseased Fine-Tuning

The downstream task fine-tunes the phase-2 gut-adapted model on `_prev3` gut data and evaluates on the independent ExVal cohort. Healthy samples are labeled H and every non-healthy phenotype is labeled D. The main metric is macro-F1 because the external validation set is imbalanced and macro-F1 gives equal importance to H and D performance.

The pipeline reports accuracy, F1, AUROC, macro-accuracy, macro-F1, macro-AUROC, H accuracy, and D accuracy. H accuracy is the true-negative rate for Healthy samples. D accuracy is the true-positive rate for Diseased samples.

## Class Imbalance and Synthetic Diseased Samples

The training set contains more Healthy than Diseased samples. The pipeline augments the Diseased class by adding Gaussian noise to nonzero abundance entries only, clips negative values to zero, and preserves zero-abundance species as zero. This follows the paper's augmentation principle and avoids inventing species that were absent in a real sample.

## Biomedical Interpretation

Good ExVal macro-F1 would suggest that the pretrained representation transfers across study cohorts and captures disease-associated microbial configurations robust to dataset shift. Per-class accuracy is essential: high overall accuracy can hide poor Diseased recall when Healthy samples dominate. Species UMAPs and attention/embedding analyses can then identify whether performance is driven by broad ecological signals, taxonomic families, or disease-enriched species groups.

The gut vs non-gut sample UMAP is a sanity check for ecological representation learning. If stool samples separate from oral, skin, and vaginal samples without body-site labels during pretraining, the model has learned high-level microbiome niche structure. The taxonomy-colored species UMAP is a sanity check for biological organization in token space. If species from the same genus, family, or order cluster, the hierarchy embeddings are shaping the latent space in a way that should improve interpretability and may improve transfer to species-sparse external cohorts.

## Current Implementation Status

Implemented in `dataset_v3/biomegpt_taxonomy_pipeline.py`:

- taxonomy-aware BiomeGPT model
- rank-wise embedding composition
- species prompt extraction
- sample prompt extraction
- phase-1/phase-2 pretraining command
- UMAP/PCA export for samples and species
- Healthy vs Diseased fine-tuning
- Diseased-class synthetic augmentation with zero-preserving noise
- optional L1/Lasso-style feature selection
- threshold optimization over 0.1 to 0.9
- ExVal metrics and prediction export

## Recommended Commands

Install optional dependencies:

```powershell
pip install -r dataset_v3\\requirements_taxonomy_pipeline.txt
```

Pretrain taxonomy-aware model:

```powershell
python dataset_v3\\biomegpt_taxonomy_pipeline.py pretrain --data_dir dataset_v3 --taxonomy_xlsx species_taxonomy_filled_validated_Serena.xlsx --output_dir dataset_v3\\outputs_taxonomy_biomegpt --epochs_phase1 30 --epochs_phase2 10 --batch_size 64 --mixed_precision
```

Extract sample and species UMAPs:

```powershell
python dataset_v3\\biomegpt_taxonomy_pipeline.py embeddings --data_dir dataset_v3 --taxonomy_xlsx species_taxonomy_filled_validated_Serena.xlsx --checkpoint dataset_v3\\outputs_taxonomy_biomegpt\\taxonomy_checkpoint_stage2.pt --output_dir dataset_v3\\outputs_taxonomy_biomegpt --sample_umap --species_umap
```

Fine-tune Healthy vs Diseased and evaluate ExVal:

```powershell
python dataset_v3\\biomegpt_taxonomy_pipeline.py finetune_hd --data_dir dataset_v3 --exval_dir ExVal --taxonomy_xlsx species_taxonomy_filled_validated_Serena.xlsx --checkpoint dataset_v3\\outputs_taxonomy_biomegpt\\taxonomy_checkpoint_stage2.pt --output_dir dataset_v3\\outputs_taxonomy_biomegpt --epochs 20 --batch_size 64 --augment_diseased --synthetic_std 5 --use_l1_feature_selection
```
"""
    out.write_text(text, encoding="utf-8")
    print(f"Wrote report: {out}")


def write_professor_markdown_artifacts(output_dir: Path, smoke_test: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "SMOKE TEST" if smoke_test else "FULL RUN"
    metrics_path = output_dir / "exval_hd_metrics.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    ex_metrics = metrics_payload.get("exval_metrics", {})
    contract_path = output_dir / "data_contract_summary.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.exists() else {}
    sanity_path = output_dir / "exval_metric_sanity_checks.json"
    sanity = json.loads(sanity_path.read_text(encoding="utf-8")) if sanity_path.exists() else {}

    def write(name: str, text: str) -> None:
        (output_dir / name).write_text(text.strip() + "\n", encoding="utf-8")

    write(
        "pipeline_status.md",
        f"""
# Pipeline Status

Run mode: **{mode}**

Implemented stages:
- Data alignment and contracts
- Taxonomy cleaning and rank-wise encoding
- Species prompt extraction
- Sample prompt extraction
- Representation analysis with plots and quantitative summaries
- Healthy vs Diseased fine-tuning from the phase-2 checkpoint
- Training-side threshold optimization for macro-F1
- ExVal evaluation and report artifacts

Data contract snapshot:
- Phase1 species: {contract.get('phase1_species_count', 'not available')}
- Phase2 species: {contract.get('phase2_species_count', 'not available')}
- `_prev3` species: {contract.get('prev3_species_count', 'not available')}
- ExVal species: {contract.get('exval_species_count', 'not available')}
- `_prev3` species missing from ExVal: {contract.get('prev3_species_missing_from_exval', 'not available')}

Interpretation note: smoke-test outputs verify code paths and file contracts only. They are not scientific performance estimates.
""",
    )
    write(
        "bugs_fixed.md",
        """
# Bugs Fixed and Safeguards Added

- Prevented notebook argparse auto-execution in Colab.
- Fixed notebook checklist cell that contained literal `\\n` strings.
- Added stratified training-side calibration split for epoch and threshold selection.
- Applied Diseased-class synthetic augmentation during selection as well as final training.
- Restricted optional L1 feature selection to the selection-training split.
- Added confusion matrix, class-specific accuracy checks, AUROC/probability sanity checks, and probability histogram.
- Added species alignment, taxonomy completeness, and label-balance artifacts.
""",
    )
    write(
        "results_summary.md",
        f"""
# Results Summary

Run mode: **{mode}**

ExVal metrics:
- Accuracy: {ex_metrics.get('accuracy', 'not available')}
- F1: {ex_metrics.get('f1', 'not available')}
- AUROC: {ex_metrics.get('auroc', 'not available')}
- Macro-accuracy: {ex_metrics.get('macro_accuracy', 'not available')}
- Macro-F1: {ex_metrics.get('macro_f1', 'not available')}
- Macro-AUROC: {ex_metrics.get('macro_auroc', 'not available')}
- H accuracy: {ex_metrics.get('accuracy_h', 'not available')}
- D accuracy: {ex_metrics.get('accuracy_d', 'not available')}
- Threshold: {ex_metrics.get('threshold', 'not available')}

Sanity flags:
- All predictions one class: {sanity.get('flags', {}).get('all_predictions_one_class', 'not available')}
- Low probability spread: {sanity.get('flags', {}).get('low_probability_spread', 'not available')}
- Class accuracy formulas verified: {sanity.get('class_accuracy_formula_check', {}).get('matches_metrics', 'not available')}
- AUROC uses non-binary probabilities: {sanity.get('auroc_uses_probabilities', 'not available')}

Scientific interpretation: use full-run outputs for conclusions. Smoke-test results only show that the pipeline executes and diagnostics are produced.
""",
    )
    write(
        "next_steps.md",
        """
# Next Steps

1. Run the notebook with `SMOKE_TEST = False` on Colab.
2. Inspect `exval_hd_metrics.json`, `exval_confusion_matrix.csv`, and `exval_probability_histogram.png`.
3. Compare default run with `USE_L1_FEATURE_SELECTION = True`.
4. Use macro-F1 as the primary model-selection metric.
5. Treat UMAPs as exploratory and cite kNN purity / sample separation summaries as quantitative support.
6. If ExVal predictions collapse to one class in a full run, inspect threshold, probability histogram, calibration split balance, and class-specific accuracies before reporting.
""",
    )


def add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--taxonomy_xlsx", default="species_taxonomy_filled_validated_Serena.xlsx")
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taxonomy-aware BiomeGPT workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pretrain")
    add_common_model_args(p)
    p.add_argument("--data_dir", default="dataset_v3")
    p.add_argument("--output_dir", default="dataset_v3/outputs_taxonomy_biomegpt")
    p.add_argument("--epochs_phase1", type=int, default=30)
    p.add_argument("--epochs_phase2", type=int, default=10)
    p.add_argument("--max_phase1_samples", type=int, default=0)
    p.add_argument("--max_phase2_samples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--mixed_precision", action="store_true")

    p = sub.add_parser("embeddings")
    add_common_model_args(p)
    p.add_argument("--data_dir", default="dataset_v3")
    p.add_argument("--output_dir", default="dataset_v3/outputs_taxonomy_biomegpt")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--sample_umap", action="store_true")
    p.add_argument("--species_umap", action="store_true")

    p = sub.add_parser("finetune_hd")
    add_common_model_args(p)
    p.add_argument("--data_dir", default="dataset_v3")
    p.add_argument("--exval_dir", default="ExVal")
    p.add_argument("--output_dir", default="dataset_v3/outputs_taxonomy_biomegpt")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--selection_fraction", type=float, default=0.1)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_exval_samples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--augment_diseased", action="store_true")
    p.add_argument("--synthetic_std", type=float, default=5.0)
    p.add_argument("--use_l1_feature_selection", action="store_true")

    p = sub.add_parser("write_report")
    p.add_argument("--output", default="dataset_v3/outputs_taxonomy_biomegpt/taxonomy_biomegpt_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "pretrain":
        run_pretrain(args)
    elif args.command == "embeddings":
        run_embeddings(args)
    elif args.command == "finetune_hd":
        run_finetune(args)
    elif args.command == "write_report":
        write_report(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
