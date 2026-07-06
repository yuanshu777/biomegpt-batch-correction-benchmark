from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.grl_correction.grl import gradient_reverse
from src.grl_correction.nomean_adapter import (
    class_weights,
    conditional_coral_loss,
    effective_rank,
    encode_labels,
    make_lambda,
    pc1_condition_auc,
    standardize_matrix,
    variance_loss,
)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for split CLS adapter training.") from exc
    return torch


def _nn():
    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for split CLS adapter training.") from exc
    return nn


@dataclass
class SplitAdapterConfig:
    inv_dim: int = 128
    nuisance_dim: int = 64
    hidden_dim: int = 256
    study_embedding_dim: int = 16
    dropout: float = 0.05
    epochs: int = 120
    warmup_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_grl: float = 0.5
    lambda_schedule: str = "linear"
    reconstruction_weight: float = 1.0
    adversary_weight: float = 0.1
    distance_weight: float = 0.05
    variance_weight: float = 0.05
    conditional_coral_weight: float = 0.0
    use_class_weights: bool = True
    seed: int = 42


@dataclass
class SplitAdapterResult:
    corrected_embeddings: pd.DataFrame
    nuisance_embeddings: pd.DataFrame
    training_history: pd.DataFrame
    diagnostics: dict[str, Any]
    label_maps: dict[str, dict[str, int]]
    config: dict[str, Any]


class SplitCLSAdapter(_nn().Module):
    def __init__(
        self,
        input_dim: int,
        n_studies: int,
        n_conditions: int,
        inv_dim: int,
        nuisance_dim: int,
        hidden_dim: int,
        study_embedding_dim: int,
        dropout: float,
    ) -> None:
        nn = _nn()
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.inv_head = nn.Linear(hidden_dim, inv_dim)
        self.nuisance_head = nn.Linear(hidden_dim, nuisance_dim)
        self.study_embedding = nn.Embedding(n_studies, study_embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(inv_dim + nuisance_dim + study_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.study_adversary = nn.Sequential(
            nn.Linear(inv_dim + n_conditions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_studies),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        return self.inv_head(hidden), self.nuisance_head(hidden)

    def forward(self, x, study_idx, condition_onehot, lambda_grl: float):
        torch = _torch()
        z_inv, z_nuisance = self.encode(x)
        reconstruction = self.decoder(torch.cat([z_inv, z_nuisance, self.study_embedding(study_idx)], dim=1))
        adversary_input = torch.cat([gradient_reverse(z_inv, lambda_grl), condition_onehot.detach()], dim=1)
        study_logits = self.study_adversary(adversary_input)
        return z_inv, z_nuisance, reconstruction, study_logits


def pairwise_distance_preservation_loss(z, x):
    torch = _torch()
    if z.shape[0] < 3:
        return torch.tensor(0.0, device=z.device)
    dz = torch.cdist(z, z)
    dx = torch.cdist(x, x).detach()
    dz = dz / (dz.mean().detach() + 1e-6)
    dx = dx / (dx.mean().detach() + 1e-6)
    return torch.mean((dz - dx) ** 2)


def train_split_cls_adapter(
    embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
    config: SplitAdapterConfig | None = None,
) -> SplitAdapterResult:
    torch = _torch()
    nn = _nn()
    config = config or SplitAdapterConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    metadata = metadata.copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    matrix = embeddings.copy()
    matrix["sample_id"] = matrix["sample_id"].astype(str)
    matrix = matrix.set_index("sample_id").loc[metadata["sample_id"].tolist()].reset_index()
    x_np, _, _, _, sample_ids = standardize_matrix(matrix)
    study_idx_np, study_map = encode_labels(metadata["studyID"])
    condition_idx_np, condition_map = encode_labels(metadata["study_condition"])
    n_conditions = len(condition_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device)
    study_tensor = torch.tensor(study_idx_np, dtype=torch.long, device=device)
    condition_tensor = torch.tensor(condition_idx_np, dtype=torch.long, device=device)
    condition_onehot = torch.nn.functional.one_hot(condition_tensor, num_classes=n_conditions).float()

    dataset = torch.utils.data.TensorDataset(x_tensor, study_tensor, condition_tensor, condition_onehot)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    model = SplitCLSAdapter(
        input_dim=x_np.shape[1],
        n_studies=len(study_map),
        n_conditions=n_conditions,
        inv_dim=config.inv_dim,
        nuisance_dim=config.nuisance_dim,
        hidden_dim=config.hidden_dim,
        study_embedding_dim=config.study_embedding_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ce_weight = class_weights(study_idx_np).to(device) if config.use_class_weights else None
    study_loss_fn = nn.CrossEntropyLoss(weight=ce_weight)
    mse = nn.MSELoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        lambda_value = make_lambda(epoch, config)  # type: ignore[arg-type]
        totals = {
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "adversary_loss": 0.0,
            "distance_loss": 0.0,
            "variance_loss": 0.0,
            "conditional_coral_loss": 0.0,
        }
        n_seen = 0
        for xb, study_b, condition_idx_b, condition_b in loader:
            optimizer.zero_grad(set_to_none=True)
            z_inv, _, reconstruction, study_logits = model(xb, study_b, condition_b, lambda_value)
            recon_loss = mse(reconstruction, xb)
            adv_loss = study_loss_fn(study_logits, study_b)
            dist_loss = pairwise_distance_preservation_loss(z_inv, xb)
            var_loss = variance_loss(z_inv)
            coral_loss = (
                conditional_coral_loss(z_inv, study_b, condition_idx_b)
                if config.conditional_coral_weight > 0.0
                else torch.tensor(0.0, device=device)
            )
            loss = (
                config.reconstruction_weight * recon_loss
                + config.adversary_weight * adv_loss
                + config.distance_weight * dist_loss
                + config.variance_weight * var_loss
                + config.conditional_coral_weight * coral_loss
            )
            loss.backward()
            optimizer.step()
            batch_n = int(xb.shape[0])
            n_seen += batch_n
            totals["total_loss"] += float(loss.detach().cpu()) * batch_n
            totals["reconstruction_loss"] += float(recon_loss.detach().cpu()) * batch_n
            totals["adversary_loss"] += float(adv_loss.detach().cpu()) * batch_n
            totals["distance_loss"] += float(dist_loss.detach().cpu()) * batch_n
            totals["variance_loss"] += float(var_loss.detach().cpu()) * batch_n
            totals["conditional_coral_loss"] += float(coral_loss.detach().cpu()) * batch_n
        row = {"epoch": float(epoch), "lambda_grl": float(lambda_value)}
        row.update({key: value / max(1, n_seen) for key, value in totals.items()})
        history.append(row)

    model.eval()
    with torch.no_grad():
        z_inv, z_nuisance = model.encode(x_tensor)
        z_inv_np = z_inv.detach().cpu().numpy()
        z_nuisance_np = z_nuisance.detach().cpu().numpy()
    corrected = pd.DataFrame(z_inv_np, columns=[f"z_inv_{i}" for i in range(z_inv_np.shape[1])])
    corrected.insert(0, "sample_id", sample_ids)
    nuisance = pd.DataFrame(z_nuisance_np, columns=[f"z_nuisance_{i}" for i in range(z_nuisance_np.shape[1])])
    nuisance.insert(0, "sample_id", sample_ids)
    diagnostics = {
        "effective_rank": effective_rank(z_inv_np),
        "raw_effective_rank": effective_rank(x_np),
        "pc1_condition_auc": pc1_condition_auc(z_inv_np, metadata),
        "raw_pc1_condition_auc": pc1_condition_auc(x_np, metadata),
        "mean_squared_shift_standardized": float("nan"),
        "mean_l2_shift_standardized": float("nan"),
        "output_dim": int(z_inv_np.shape[1]),
    }
    return SplitAdapterResult(
        corrected_embeddings=corrected,
        nuisance_embeddings=nuisance,
        training_history=pd.DataFrame(history),
        diagnostics=diagnostics,
        label_maps={"study_labels": study_map, "condition_labels": condition_map},
        config=asdict(config),
    )
