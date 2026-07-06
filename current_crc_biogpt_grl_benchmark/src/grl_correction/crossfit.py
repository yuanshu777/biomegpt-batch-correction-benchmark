from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .adversary import MLPAdversary
from .conditioned_decoder import StudyConditionedDecoder
from .grl import gradient_reverse
from .train_grl import _class_weights, _make_corrected_frame, grl_lambda_for_epoch


@dataclass
class EarlyStoppingConfig:
    sample_id_column: str = "sample_id"
    study_column: str = "studyID"
    condition_column: str = "study_condition"
    positive_condition: str = "CRC"
    latent_dim: int = 8
    hidden_dim: int = 128
    epochs: int = 100
    lr: float = 1e-3
    lambda_grl: float = 10.0
    lambda_schedule: str = "linear"
    warmup_fraction: float = 0.1
    condition_weight: float = 0.1
    preserve_weight: float = 0.001
    use_class_weights: bool = True
    batch_size: int = 64
    condition_aware_adversary: bool = True
    condition_embedding_dim: int = 16
    use_study_conditioned_decoder: bool = False
    eval_every: int = 5
    condition_auc_margin: float = 0.05
    patience_evals: int = 6
    seed: int = 42
    device: str = "cpu"


@dataclass
class EarlyStoppingResult:
    corrected_embeddings: pd.DataFrame
    trace: pd.DataFrame
    selected: dict[str, Any]
    config: dict[str, Any]


class GRLEncoderBundle:
    def __init__(
        self,
        encoder,
        mean: np.ndarray,
        std: np.ndarray,
        feature_cols: list[str],
        sample_id_column: str,
        device: str,
    ):
        self.encoder = encoder
        self.mean = mean
        self.std = std
        self.feature_cols = feature_cols
        self.sample_id_column = sample_id_column
        self.device = device

    def transform(self, matrix: pd.DataFrame) -> pd.DataFrame:
        import torch

        sample_ids = matrix[self.sample_id_column].astype(str).tolist()
        x_np = matrix[self.feature_cols].astype(float).to_numpy(dtype=np.float32)
        x_std = ((x_np - self.mean) / self.std).astype(np.float32)
        x = torch.tensor(x_std, dtype=torch.float32, device=torch.device(self.device))
        self.encoder.eval()
        with torch.no_grad():
            z_np = self.encoder(x).detach().cpu().numpy()
        return _make_corrected_frame(sample_ids, z_np, self.sample_id_column)


def _standardize_train(x_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_np.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_np.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ((x_np - mean) / std).astype(np.float32), mean, std


def _subset_matrix(matrix: pd.DataFrame, sample_ids: set[str], sample_id_column: str = "sample_id") -> pd.DataFrame:
    return matrix[matrix[sample_id_column].astype(str).isin(sample_ids)].copy()


def _evaluate_probes(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    config: EarlyStoppingConfig,
) -> dict[str, float]:
    from src.evaluation.probes import probe_classification

    study = probe_classification(
        matrix,
        metadata,
        config.study_column,
        sample_id_column=config.sample_id_column,
        random_state=config.seed,
    )
    condition = probe_classification(
        matrix,
        metadata,
        config.condition_column,
        sample_id_column=config.sample_id_column,
        positive_label=config.positive_condition,
        random_state=config.seed,
    )
    return {
        "study_balanced_accuracy": float(study["balanced_accuracy"]),
        "study_macro_f1": float(study["macro_f1"]),
        "condition_auc": float(condition.get("auroc", np.nan)),
        "condition_balanced_accuracy": float(condition["balanced_accuracy"]),
        "condition_macro_f1": float(condition["macro_f1"]),
    }


def _select_checkpoint(trace: pd.DataFrame, condition_auc_margin: float) -> dict[str, Any]:
    scored = trace.copy()
    scored["condition_floor"] = scored["raw_validation_condition_auc"] - condition_auc_margin
    scored["constraint_satisfied"] = scored["validation_condition_auc"] >= scored["condition_floor"]
    scored["condition_shortfall"] = (scored["condition_floor"] - scored["validation_condition_auc"]).clip(lower=0)
    scored["tradeoff_score"] = scored["validation_study_balanced_accuracy"] + 5.0 * scored["condition_shortfall"]

    valid = scored[scored["constraint_satisfied"]]
    if not valid.empty:
        row = valid.sort_values(
            ["validation_study_balanced_accuracy", "validation_condition_auc"],
            ascending=[True, False],
        ).iloc[0]
        reason = "constraint_satisfied_lowest_validation_study_ba"
    else:
        row = scored.sort_values(
            ["tradeoff_score", "validation_study_balanced_accuracy", "validation_condition_auc"],
            ascending=[True, True, False],
        ).iloc[0]
        reason = "constraint_failed_best_tradeoff"
    out = row.to_dict()
    out["selection_reason"] = reason
    return out


def train_grl_with_validation_early_stopping(
    train_matrix: pd.DataFrame,
    train_metadata: pd.DataFrame,
    validation_matrix: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    transform_matrix: pd.DataFrame,
    config: EarlyStoppingConfig | None = None,
    fold_id: str | int = "full",
) -> EarlyStoppingResult:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    config = config or EarlyStoppingConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)

    joined = train_metadata[[config.sample_id_column, config.study_column, config.condition_column]].merge(
        train_matrix,
        on=config.sample_id_column,
        how="inner",
    )
    if joined.empty:
        raise ValueError("No aligned train samples for GRL early stopping.")
    feature_cols = [c for c in joined.columns if c not in {config.sample_id_column, config.study_column, config.condition_column}]
    x_np = joined[feature_cols].astype(float).to_numpy(dtype=np.float32)
    x_std_np, mean, std = _standardize_train(x_np)

    study_values = sorted(joined[config.study_column].astype(str).unique())
    condition_values = sorted(joined[config.condition_column].astype(str).unique())
    study_map = {label: i for i, label in enumerate(study_values)}
    condition_map = {label: i for i, label in enumerate(condition_values)}
    y_study_np = np.array([study_map[v] for v in joined[config.study_column].astype(str)], dtype=np.int64)
    y_condition_np = np.array([condition_map[v] for v in joined[config.condition_column].astype(str)], dtype=np.int64)

    x = torch.tensor(x_std_np, dtype=torch.float32, device=device)
    y_study = torch.tensor(y_study_np, dtype=torch.long, device=device)
    y_condition = torch.tensor(y_condition_np, dtype=torch.long, device=device)

    input_dim = int(x.shape[1])
    latent_dim = min(int(config.latent_dim), input_dim)
    encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.LayerNorm(latent_dim)).to(device)
    simple_decoder = None
    study_decoder = None
    if config.use_study_conditioned_decoder:
        study_decoder = StudyConditionedDecoder(latent_dim, len(study_map), input_dim, hidden_dim=config.hidden_dim).to(device)
    elif latent_dim != input_dim:
        simple_decoder = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, input_dim),
        ).to(device)

    condition_embedding = None
    study_adv_input_dim = latent_dim
    if config.condition_aware_adversary:
        condition_embedding = nn.Embedding(len(condition_map), config.condition_embedding_dim).to(device)
        study_adv_input_dim += config.condition_embedding_dim

    study_adv = MLPAdversary(study_adv_input_dim, len(study_map), hidden_dim=config.hidden_dim).to(device)
    condition_head = MLPAdversary(latent_dim, len(condition_map), hidden_dim=config.hidden_dim).to(device)

    modules = [encoder, study_adv, condition_head]
    if simple_decoder is not None:
        modules.append(simple_decoder)
    if study_decoder is not None:
        modules.append(study_decoder)
    if condition_embedding is not None:
        modules.append(condition_embedding)
    optimizer = torch.optim.Adam([p for module in modules for p in module.parameters()], lr=config.lr)

    study_weights = _class_weights(y_study_np, len(study_map), device, config.use_class_weights)
    condition_weights = _class_weights(y_condition_np, len(condition_map), device, config.use_class_weights)
    loader = DataLoader(
        TensorDataset(x, y_study, y_condition),
        batch_size=min(int(config.batch_size), len(x)),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(config.seed),
    )

    raw_validation_metrics = _evaluate_probes(validation_matrix, validation_metadata, config)
    raw_validation_condition_auc = raw_validation_metrics["condition_auc"]
    trace_rows: list[dict[str, Any]] = []
    best_state = copy.deepcopy(encoder.state_dict())
    best_score = float("inf")
    stale_evals = 0

    for epoch in range(1, config.epochs + 1):
        encoder.train()
        study_adv.train()
        condition_head.train()
        if simple_decoder is not None:
            simple_decoder.train()
        if study_decoder is not None:
            study_decoder.train()

        current_lambda = grl_lambda_for_epoch(
            epoch,
            config.epochs,
            config.lambda_grl,
            config.lambda_schedule,
            config.warmup_fraction,
        )
        batch_rows = []
        for batch_x, batch_study, batch_condition in loader:
            optimizer.zero_grad(set_to_none=True)
            z = encoder(batch_x)
            z_for_adv = gradient_reverse(z, current_lambda)
            if condition_embedding is not None:
                adv_input = torch.cat([z_for_adv, condition_embedding(batch_condition)], dim=1)
            else:
                adv_input = z_for_adv
            study_loss = F.cross_entropy(study_adv(adv_input), batch_study, weight=study_weights)
            condition_loss = F.cross_entropy(condition_head(z), batch_condition, weight=condition_weights)
            if study_decoder is not None:
                recon = study_decoder(z, batch_study)
                preservation_loss = F.mse_loss(recon, batch_x)
            elif simple_decoder is not None:
                recon = simple_decoder(z)
                preservation_loss = F.mse_loss(recon, batch_x)
            else:
                preservation_loss = F.mse_loss(z, batch_x)
            loss = study_loss + config.condition_weight * condition_loss + config.preserve_weight * preservation_loss
            loss.backward()
            optimizer.step()
            batch_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "study_adversary_loss": float(study_loss.detach().cpu()),
                    "condition_loss": float(condition_loss.detach().cpu()),
                    "preservation_loss": float(preservation_loss.detach().cpu()),
                    "n_batch_samples": int(batch_x.shape[0]),
                }
            )

        if epoch % int(config.eval_every) != 0 and epoch != config.epochs:
            continue

        bundle = GRLEncoderBundle(encoder, mean, std, feature_cols, config.sample_id_column, config.device)
        validation_z = bundle.transform(validation_matrix)
        validation_metrics = _evaluate_probes(validation_z, validation_metadata, config)
        row = {
            "fold_id": fold_id,
            "epoch": epoch,
            "lambda_grl": current_lambda,
            "raw_validation_condition_auc": raw_validation_condition_auc,
            "validation_study_balanced_accuracy": validation_metrics["study_balanced_accuracy"],
            "validation_study_macro_f1": validation_metrics["study_macro_f1"],
            "validation_condition_auc": validation_metrics["condition_auc"],
            "validation_condition_balanced_accuracy": validation_metrics["condition_balanced_accuracy"],
            "validation_condition_macro_f1": validation_metrics["condition_macro_f1"],
            "loss": float(np.average([r["loss"] for r in batch_rows], weights=[r["n_batch_samples"] for r in batch_rows])),
            "study_adversary_loss": float(
                np.average([r["study_adversary_loss"] for r in batch_rows], weights=[r["n_batch_samples"] for r in batch_rows])
            ),
            "condition_loss": float(
                np.average([r["condition_loss"] for r in batch_rows], weights=[r["n_batch_samples"] for r in batch_rows])
            ),
            "preservation_loss": float(
                np.average([r["preservation_loss"] for r in batch_rows], weights=[r["n_batch_samples"] for r in batch_rows])
            ),
        }
        condition_floor = row["raw_validation_condition_auc"] - config.condition_auc_margin
        condition_shortfall = max(0.0, condition_floor - row["validation_condition_auc"])
        tradeoff_score = row["validation_study_balanced_accuracy"] + 5.0 * condition_shortfall
        trace_rows.append(row)
        if tradeoff_score < best_score - 1e-8:
            best_score = tradeoff_score
            best_state = copy.deepcopy(encoder.state_dict())
            stale_evals = 0
        else:
            stale_evals += 1
        if stale_evals >= int(config.patience_evals):
            break

    trace = pd.DataFrame(trace_rows)
    selected = _select_checkpoint(trace, config.condition_auc_margin)
    encoder.load_state_dict(best_state)
    selected_epoch = int(selected["epoch"])
    for row in trace_rows:
        if int(row["epoch"]) == selected_epoch:
            selected_state_epoch = row["epoch"]
            break
    else:
        selected_state_epoch = selected_epoch

    # Refit selected state according to the explicit selection rule, not only the
    # early stopping patience tracker. This keeps the chosen checkpoint auditable.
    # If the selected epoch differs from best_state, retraining is cheap and
    # deterministic for this small local benchmark.
    if int(selected_state_epoch) != int(trace.iloc[trace["validation_study_balanced_accuracy"].idxmin()]["epoch"]):
        pass

    # The patience tracker may have kept a tradeoff-best state before the final
    # selection dataframe was known. Re-train to the selected epoch for exactness.
    retrained = _train_to_epoch(
        train_matrix,
        train_metadata,
        config,
        feature_cols,
        study_map,
        condition_map,
        mean,
        std,
        int(selected_epoch),
    )
    bundle = GRLEncoderBundle(retrained, mean, std, feature_cols, config.sample_id_column, config.device)
    corrected = bundle.transform(transform_matrix)
    selected["fold_id"] = fold_id
    selected["n_train_samples"] = int(len(train_matrix))
    selected["n_validation_samples"] = int(len(validation_matrix))
    selected["n_transform_samples"] = int(len(transform_matrix))
    return EarlyStoppingResult(
        corrected_embeddings=corrected,
        trace=trace,
        selected=selected,
        config=asdict(config),
    )


def _train_to_epoch(
    train_matrix: pd.DataFrame,
    train_metadata: pd.DataFrame,
    config: EarlyStoppingConfig,
    feature_cols: list[str],
    study_map: dict[str, int],
    condition_map: dict[str, int],
    mean: np.ndarray,
    std: np.ndarray,
    selected_epoch: int,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    joined = train_metadata[[config.sample_id_column, config.study_column, config.condition_column]].merge(
        train_matrix,
        on=config.sample_id_column,
        how="inner",
    )
    x_np = joined[feature_cols].astype(float).to_numpy(dtype=np.float32)
    x_std_np = ((x_np - mean) / std).astype(np.float32)
    y_study_np = np.array([study_map[v] for v in joined[config.study_column].astype(str)], dtype=np.int64)
    y_condition_np = np.array([condition_map[v] for v in joined[config.condition_column].astype(str)], dtype=np.int64)
    x = torch.tensor(x_std_np, dtype=torch.float32, device=device)
    y_study = torch.tensor(y_study_np, dtype=torch.long, device=device)
    y_condition = torch.tensor(y_condition_np, dtype=torch.long, device=device)

    input_dim = int(x.shape[1])
    latent_dim = min(int(config.latent_dim), input_dim)
    encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.LayerNorm(latent_dim)).to(device)
    simple_decoder = None
    study_decoder = None
    if config.use_study_conditioned_decoder:
        study_decoder = StudyConditionedDecoder(latent_dim, len(study_map), input_dim, hidden_dim=config.hidden_dim).to(device)
    elif latent_dim != input_dim:
        simple_decoder = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, input_dim),
        ).to(device)

    condition_embedding = None
    study_adv_input_dim = latent_dim
    if config.condition_aware_adversary:
        condition_embedding = nn.Embedding(len(condition_map), config.condition_embedding_dim).to(device)
        study_adv_input_dim += config.condition_embedding_dim
    study_adv = MLPAdversary(study_adv_input_dim, len(study_map), hidden_dim=config.hidden_dim).to(device)
    condition_head = MLPAdversary(latent_dim, len(condition_map), hidden_dim=config.hidden_dim).to(device)
    modules = [encoder, study_adv, condition_head]
    if simple_decoder is not None:
        modules.append(simple_decoder)
    if study_decoder is not None:
        modules.append(study_decoder)
    if condition_embedding is not None:
        modules.append(condition_embedding)
    optimizer = torch.optim.Adam([p for module in modules for p in module.parameters()], lr=config.lr)
    study_weights = _class_weights(y_study_np, len(study_map), device, config.use_class_weights)
    condition_weights = _class_weights(y_condition_np, len(condition_map), device, config.use_class_weights)
    loader = DataLoader(
        TensorDataset(x, y_study, y_condition),
        batch_size=min(int(config.batch_size), len(x)),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(config.seed),
    )
    for epoch in range(1, selected_epoch + 1):
        current_lambda = grl_lambda_for_epoch(
            epoch,
            config.epochs,
            config.lambda_grl,
            config.lambda_schedule,
            config.warmup_fraction,
        )
        for batch_x, batch_study, batch_condition in loader:
            optimizer.zero_grad(set_to_none=True)
            z = encoder(batch_x)
            z_for_adv = gradient_reverse(z, current_lambda)
            if condition_embedding is not None:
                adv_input = torch.cat([z_for_adv, condition_embedding(batch_condition)], dim=1)
            else:
                adv_input = z_for_adv
            study_loss = F.cross_entropy(study_adv(adv_input), batch_study, weight=study_weights)
            condition_loss = F.cross_entropy(condition_head(z), batch_condition, weight=condition_weights)
            if study_decoder is not None:
                recon = study_decoder(z, batch_study)
                preservation_loss = F.mse_loss(recon, batch_x)
            elif simple_decoder is not None:
                recon = simple_decoder(z)
                preservation_loss = F.mse_loss(recon, batch_x)
            else:
                preservation_loss = F.mse_loss(z, batch_x)
            loss = study_loss + config.condition_weight * condition_loss + config.preserve_weight * preservation_loss
            loss.backward()
            optimizer.step()
    return encoder


def stratified_sample_folds(metadata: pd.DataFrame, n_splits: int, seed: int, sample_id_column: str = "sample_id") -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold

    labels = metadata["studyID"].astype(str) + "||" + metadata["study_condition"].astype(str)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(metadata))
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(idx, labels)]


def split_train_validation(
    metadata: pd.DataFrame,
    validation_fraction: float,
    seed: int,
    sample_id_column: str = "sample_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import StratifiedShuffleSplit

    labels = metadata["studyID"].astype(str) + "||" + metadata["study_condition"].astype(str)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.arange(len(metadata)), labels))
    return metadata.iloc[train_idx].copy(), metadata.iloc[val_idx].copy()


def align_matrix_to_metadata(matrix: pd.DataFrame, metadata: pd.DataFrame, sample_id_column: str = "sample_id") -> pd.DataFrame:
    ids = metadata[sample_id_column].astype(str).tolist()
    indexed = matrix.copy()
    indexed[sample_id_column] = indexed[sample_id_column].astype(str)
    indexed = indexed.set_index(sample_id_column).loc[ids].reset_index()
    return indexed
