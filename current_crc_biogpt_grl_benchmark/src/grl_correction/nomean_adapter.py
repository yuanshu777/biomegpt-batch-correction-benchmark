from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.grl_correction.grl import gradient_reverse


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for NoMean CLS adapter training.") from exc
    return torch


def _nn():
    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for NoMean CLS adapter training.") from exc
    return nn


@dataclass
class NoMeanAdapterConfig:
    hidden_dim: int = 128
    study_embedding_dim: int = 16
    residual_scale: float = 0.1
    dropout: float = 0.05
    epochs: int = 120
    warmup_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_grl: float = 0.1
    lambda_schedule: str = "linear"
    reconstruction_weight: float = 1.0
    preserve_weight: float = 1.0
    adversary_weight: float = 0.1
    adversary_mode: str = "conditional"
    condition_prior_weight: float = 0.1
    pairwise_weight: float = 0.0
    conditional_coral_weight: float = 0.0
    variance_weight: float = 0.02
    covariance_weight: float = 0.005
    use_class_weights: bool = True
    seed: int = 42


@dataclass
class NoMeanAdapterResult:
    corrected_embeddings: pd.DataFrame
    training_history: pd.DataFrame
    diagnostics: dict[str, Any]
    label_maps: dict[str, dict[str, int]]
    config: dict[str, Any]


class NoMeanCLSAdapter(_nn().Module):
    def __init__(
        self,
        input_dim: int,
        n_studies: int,
        n_conditions: int,
        hidden_dim: int = 128,
        study_embedding_dim: int = 16,
        residual_scale: float = 0.1,
        dropout: float = 0.05,
        adversary_mode: str = "conditional",
    ) -> None:
        nn = _nn()
        super().__init__()
        self.residual_scale = residual_scale
        self.adversary_mode = adversary_mode
        self.norm = nn.LayerNorm(input_dim)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.study_embedding = nn.Embedding(n_studies, study_embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim + study_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.vanilla_adversary = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_studies),
        )
        self.conditional_adversary = nn.Sequential(
            nn.Linear(input_dim + n_conditions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_studies),
        )
        self.condition_prior = nn.Linear(n_conditions, n_studies)
        self.residual_adversary = nn.Sequential(
            nn.Linear(input_dim + n_conditions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_studies),
        )
        self.pairwise_adversary = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x):
        return x + self.residual_scale * self.adapter(self.norm(x))

    def study_outputs(self, z, condition_onehot, lambda_grl: float) -> dict[str, Any]:
        torch = _torch()
        mode = self.adversary_mode
        z_reversed = gradient_reverse(z, lambda_grl)
        if mode == "vanilla":
            return {"study_logits": self.vanilla_adversary(z_reversed)}
        if mode == "conditional":
            adversary_input = torch.cat([z_reversed, condition_onehot.detach()], dim=1)
            return {"study_logits": self.conditional_adversary(adversary_input)}
        if mode == "residual_conditional":
            prior_logits = self.condition_prior(condition_onehot.detach())
            residual_input = torch.cat([z_reversed, condition_onehot.detach()], dim=1)
            residual_logits = self.residual_adversary(residual_input)
            return {
                "study_logits": prior_logits.detach() + residual_logits,
                "condition_prior_logits": prior_logits,
                "residual_logits": residual_logits,
            }
        if mode == "pairwise_within_condition":
            return {"study_logits": None}
        raise ValueError(f"Unsupported adversary_mode: {mode}")

    def forward(self, x, study_idx, condition_onehot, lambda_grl: float):
        z = self.encode(x)
        study_context = self.study_embedding(study_idx)
        reconstruction = self.decoder(_torch().cat([z, study_context], dim=1))
        return z, reconstruction, self.study_outputs(z, condition_onehot, lambda_grl)

    def pairwise_logits(self, z, study_idx, condition_idx, lambda_grl: float):
        torch = _torch()
        n = int(z.shape[0])
        if n < 2:
            return None, None
        row, col = torch.triu_indices(n, n, offset=1, device=z.device)
        same_condition = condition_idx[row] == condition_idx[col]
        if not bool(same_condition.any()):
            return None, None
        row = row[same_condition]
        col = col[same_condition]
        z_reversed = gradient_reverse(z, lambda_grl)
        pair_features = torch.abs(z_reversed[row] - z_reversed[col])
        same_study = (study_idx[row] == study_idx[col]).float()
        return self.pairwise_adversary(pair_features).squeeze(1), same_study


def make_lambda(epoch: int, config: NoMeanAdapterConfig) -> float:
    if epoch <= config.warmup_epochs:
        return 0.0
    effective_epoch = epoch - config.warmup_epochs
    effective_total = max(1, config.epochs - config.warmup_epochs)
    schedule = config.lambda_schedule.lower()
    if schedule == "constant":
        scale = 1.0
    elif schedule == "linear":
        scale = min(1.0, effective_epoch / effective_total)
    elif schedule == "dann":
        p = effective_epoch / effective_total
        scale = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    else:
        raise ValueError(f"Unsupported lambda_schedule: {config.lambda_schedule}")
    return float(config.lambda_grl * scale)


def standardize_matrix(matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    sample_ids = matrix["sample_id"].astype(str).tolist()
    feature_cols = [c for c in matrix.columns if c != "sample_id"]
    x = matrix[feature_cols].astype(float).to_numpy(dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32), feature_cols, sample_ids


def encode_labels(values: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    levels = sorted(values.astype(str).unique())
    mapping = {level: idx for idx, level in enumerate(levels)}
    return values.astype(str).map(mapping).to_numpy(dtype=np.int64), mapping


def class_weights(labels: np.ndarray):
    torch = _torch()
    counts = np.bincount(labels)
    weights = len(labels) / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def covariance_matrix(x):
    x = x - x.mean(dim=0, keepdim=True)
    denom = max(1, x.shape[0] - 1)
    return x.T @ x / denom


def variance_loss(z):
    torch = _torch()
    if z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return torch.relu(1.0 - std).mean()


def covariance_preservation_loss(z, x):
    torch = _torch()
    if z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device)
    return torch.mean((covariance_matrix(z) - covariance_matrix(x).detach()) ** 2)


def conditional_coral_loss(z, study_idx, condition_idx):
    torch = _torch()
    losses = []
    for condition_value in torch.unique(condition_idx):
        condition_mask = condition_idx == condition_value
        condition_studies = torch.unique(study_idx[condition_mask])
        group_tensors = []
        for study_value in condition_studies:
            group = z[condition_mask & (study_idx == study_value)]
            if group.shape[0] >= 2:
                group_tensors.append(group)
        if len(group_tensors) < 2:
            continue
        pooled = torch.cat(group_tensors, dim=0)
        pooled_mean = pooled.mean(dim=0)
        pooled_cov = covariance_matrix(pooled)
        for group in group_tensors:
            mean_loss = torch.mean((group.mean(dim=0) - pooled_mean.detach()) ** 2)
            cov_loss = torch.mean((covariance_matrix(group) - pooled_cov.detach()) ** 2)
            losses.append(mean_loss + cov_loss)
    if not losses:
        return torch.tensor(0.0, device=z.device)
    return torch.stack(losses).mean()


def effective_rank(x: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    entropy = -float(np.sum(p * np.log(p + 1e-12)))
    return float(np.exp(entropy))


def pc1_condition_auc(x: np.ndarray, metadata: pd.DataFrame) -> float:
    try:
        from sklearn.decomposition import PCA
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return float("nan")
    y = (metadata["study_condition"].astype(str).to_numpy() == "CRC").astype(int)
    pc1 = PCA(n_components=1, random_state=0).fit_transform(StandardScaler().fit_transform(x))[:, 0]
    auc = float(roc_auc_score(y, pc1))
    return max(auc, 1.0 - auc)


def train_nomean_cls_adapter(
    embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
    config: NoMeanAdapterConfig | None = None,
) -> NoMeanAdapterResult:
    torch = _torch()
    nn = _nn()
    config = config or NoMeanAdapterConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    metadata = metadata.copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    matrix = embeddings.copy()
    matrix["sample_id"] = matrix["sample_id"].astype(str)
    matrix = matrix.set_index("sample_id").loc[metadata["sample_id"].tolist()].reset_index()
    x_np, _, _, feature_cols, sample_ids = standardize_matrix(matrix)
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

    model = NoMeanCLSAdapter(
        input_dim=x_np.shape[1],
        n_studies=len(study_map),
        n_conditions=n_conditions,
        hidden_dim=config.hidden_dim,
        study_embedding_dim=config.study_embedding_dim,
        residual_scale=config.residual_scale,
        dropout=config.dropout,
        adversary_mode=config.adversary_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ce_weight = class_weights(study_idx_np).to(device) if config.use_class_weights else None
    study_loss_fn = nn.CrossEntropyLoss(weight=ce_weight)
    pairwise_loss_fn = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        lambda_value = make_lambda(epoch, config)
        totals: dict[str, float] = {
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "preservation_loss": 0.0,
            "adversary_loss": 0.0,
            "condition_prior_loss": 0.0,
            "pairwise_loss": 0.0,
            "conditional_coral_loss": 0.0,
            "variance_loss": 0.0,
            "covariance_loss": 0.0,
        }
        n_seen = 0
        for xb, study_b, condition_idx_b, condition_b in loader:
            optimizer.zero_grad(set_to_none=True)
            z, reconstruction, outputs = model(xb, study_b, condition_b, lambda_value)
            recon_loss = mse(reconstruction, xb)
            preserve_loss = mse(z, xb)
            if outputs.get("study_logits") is None:
                adv_loss = torch.tensor(0.0, device=device)
            else:
                adv_loss = study_loss_fn(outputs["study_logits"], study_b)
            if outputs.get("condition_prior_logits") is None:
                prior_loss = torch.tensor(0.0, device=device)
            else:
                prior_loss = study_loss_fn(outputs["condition_prior_logits"], study_b)
            pair_logits, pair_target = model.pairwise_logits(z, study_b, condition_idx_b, lambda_value)
            if pair_logits is None or config.pairwise_weight == 0.0:
                pair_loss = torch.tensor(0.0, device=device)
            else:
                pair_loss = pairwise_loss_fn(pair_logits, pair_target)
            if config.conditional_coral_weight == 0.0:
                coral_loss = torch.tensor(0.0, device=device)
            else:
                coral_loss = conditional_coral_loss(z, study_b, condition_idx_b)
            var_loss = variance_loss(z)
            cov_loss = covariance_preservation_loss(z, xb)
            loss = (
                config.reconstruction_weight * recon_loss
                + config.preserve_weight * preserve_loss
                + config.adversary_weight * adv_loss
                + config.condition_prior_weight * prior_loss
                + config.pairwise_weight * pair_loss
                + config.conditional_coral_weight * coral_loss
                + config.variance_weight * var_loss
                + config.covariance_weight * cov_loss
            )
            loss.backward()
            optimizer.step()
            batch_n = int(xb.shape[0])
            n_seen += batch_n
            totals["total_loss"] += float(loss.detach().cpu()) * batch_n
            totals["reconstruction_loss"] += float(recon_loss.detach().cpu()) * batch_n
            totals["preservation_loss"] += float(preserve_loss.detach().cpu()) * batch_n
            totals["adversary_loss"] += float(adv_loss.detach().cpu()) * batch_n
            totals["condition_prior_loss"] += float(prior_loss.detach().cpu()) * batch_n
            totals["pairwise_loss"] += float(pair_loss.detach().cpu()) * batch_n
            totals["conditional_coral_loss"] += float(coral_loss.detach().cpu()) * batch_n
            totals["variance_loss"] += float(var_loss.detach().cpu()) * batch_n
            totals["covariance_loss"] += float(cov_loss.detach().cpu()) * batch_n
        row = {"epoch": float(epoch), "lambda_grl": float(lambda_value)}
        row.update({key: value / max(1, n_seen) for key, value in totals.items()})
        history.append(row)

    model.eval()
    with torch.no_grad():
        z = model.encode(x_tensor).detach().cpu().numpy()
    corrected = pd.DataFrame(z, columns=feature_cols)
    corrected.insert(0, "sample_id", sample_ids)
    diagnostics = {
        "effective_rank": effective_rank(z),
        "raw_effective_rank": effective_rank(x_np),
        "pc1_condition_auc": pc1_condition_auc(z, metadata),
        "raw_pc1_condition_auc": pc1_condition_auc(x_np, metadata),
        "mean_squared_shift_standardized": float(np.mean((z - x_np) ** 2)),
        "mean_l2_shift_standardized": float(np.mean(np.linalg.norm(z - x_np, axis=1))),
    }
    return NoMeanAdapterResult(
        corrected_embeddings=corrected,
        training_history=pd.DataFrame(history),
        diagnostics=diagnostics,
        label_maps={"study_labels": study_map, "condition_labels": condition_map},
        config=asdict(config),
    )
