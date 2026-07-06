from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .adversary import MLPAdversary
from .conditioned_decoder import StudyConditionedDecoder
from .grl import gradient_reverse


@dataclass
class GRLTrainingConfig:
    sample_id_column: str = "sample_id"
    study_column: str = "studyID"
    condition_column: str = "study_condition"
    latent_dim: int | None = None
    hidden_dim: int = 128
    epochs: int = 25
    lr: float = 1e-3
    lambda_grl: float = 1.0
    lambda_schedule: str = "dann"
    warmup_fraction: float = 0.4
    condition_weight: float = 1.0
    preserve_weight: float = 1.0
    use_class_weights: bool = True
    batch_size: int | None = 64
    shuffle: bool = True
    seed: int = 7
    full_batch: bool = False
    condition_aware_adversary: bool = False
    condition_embedding_dim: int = 16
    use_study_conditioned_decoder: bool = False
    external_eval_every: int | None = 5
    device: str = "cpu"


@dataclass
class GRLTrainingResult:
    corrected_embeddings: pd.DataFrame
    history: pd.DataFrame
    final_probe_metrics: pd.DataFrame
    study_labels: dict[str, int]
    condition_labels: dict[str, int]
    config: dict[str, Any]


def grl_lambda_for_epoch(
    epoch: int,
    epochs: int,
    lambda_grl: float,
    lambda_schedule: str = "dann",
    warmup_fraction: float = 0.4,
) -> float:
    if epochs <= 1:
        progress = 1.0
    else:
        progress = (epoch - 1) / max(epochs - 1, 1)
    schedule = lambda_schedule.lower()
    if schedule == "constant":
        return float(lambda_grl)
    warmup_fraction = max(float(warmup_fraction), 1e-8)
    warmup_progress = min(progress / warmup_fraction, 1.0)
    if schedule == "linear":
        return float(lambda_grl * warmup_progress)
    if schedule == "dann":
        return float(lambda_grl * (2.0 / (1.0 + np.exp(-10.0 * warmup_progress)) - 1.0))
    raise ValueError(f"Unknown lambda_schedule: {lambda_schedule}")


def _class_weights(labels: np.ndarray, n_classes: int, device, enabled: bool):
    if not enabled:
        return None
    import torch

    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _standardize(x_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_np.mean(axis=0, keepdims=True)
    std = x_np.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((x_np - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _make_corrected_frame(sample_ids: list[str], z_np: np.ndarray, sample_id_column: str) -> pd.DataFrame:
    out = pd.DataFrame(z_np, columns=[f"grl_dim_{i}" for i in range(z_np.shape[1])])
    out.insert(0, sample_id_column, sample_ids)
    return out


def evaluate_external_probes(
    corrected_embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_id_column: str = "sample_id",
    study_column: str = "studyID",
    condition_column: str = "study_condition",
    positive_condition: str = "CRC",
) -> pd.DataFrame:
    try:
        from src.evaluation.probes import probe_classification
    except Exception as exc:  # pragma: no cover - environment dependent
        return pd.DataFrame(
            [
                {
                    "metric": "external_probe_status",
                    "value": "skipped",
                    "reason": f"scikit-learn is required for external probes: {exc}",
                }
            ]
        )

    rows: list[dict[str, Any]] = []
    try:
        study = probe_classification(
            corrected_embeddings,
            metadata,
            study_column,
            sample_id_column=sample_id_column,
        )
        rows.extend(
            [
                {"metric": "study_balanced_accuracy", "value": study["balanced_accuracy"], "reason": ""},
                {"metric": "study_macro_f1", "value": study["macro_f1"], "reason": ""},
            ]
        )
        condition = probe_classification(
            corrected_embeddings,
            metadata,
            condition_column,
            sample_id_column=sample_id_column,
            positive_label=positive_condition,
        )
        rows.extend(
            [
                {"metric": "condition_auroc", "value": condition.get("auroc"), "reason": ""},
                {"metric": "condition_balanced_accuracy", "value": condition["balanced_accuracy"], "reason": ""},
                {"metric": "condition_macro_f1", "value": condition["macro_f1"], "reason": ""},
            ]
        )
    except Exception as exc:
        rows.append({"metric": "external_probe_status", "value": "skipped", "reason": str(exc)})
    return pd.DataFrame(rows)


def train_embedding_grl(
    embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_id_column: str = "sample_id",
    study_column: str = "studyID",
    condition_column: str = "study_condition",
    latent_dim: int | None = None,
    hidden_dim: int = 128,
    epochs: int = 25,
    lr: float = 1e-3,
    lambda_grl: float = 1.0,
    lambda_schedule: str = "dann",
    warmup_fraction: float = 0.4,
    condition_weight: float = 1.0,
    preserve_weight: float = 1.0,
    use_class_weights: bool = True,
    batch_size: int | None = 64,
    shuffle: bool = True,
    seed: int = 7,
    full_batch: bool = False,
    condition_aware_adversary: bool = False,
    condition_embedding_dim: int = 16,
    use_study_conditioned_decoder: bool = False,
    external_eval_every: int | None = 5,
    device: str = "cpu",
) -> GRLTrainingResult:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    config = GRLTrainingConfig(
        sample_id_column=sample_id_column,
        study_column=study_column,
        condition_column=condition_column,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        lambda_grl=lambda_grl,
        lambda_schedule=lambda_schedule,
        warmup_fraction=warmup_fraction,
        condition_weight=condition_weight,
        preserve_weight=preserve_weight,
        use_class_weights=use_class_weights,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        full_batch=full_batch,
        condition_aware_adversary=condition_aware_adversary,
        condition_embedding_dim=condition_embedding_dim,
        use_study_conditioned_decoder=use_study_conditioned_decoder,
        external_eval_every=external_eval_every,
        device=device,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_device = torch.device(device)

    joined = metadata[[sample_id_column, study_column, condition_column]].merge(
        embeddings,
        on=sample_id_column,
        how="inner",
    )
    if joined.empty:
        raise ValueError("No aligned embeddings and metadata samples.")
    feature_cols = [c for c in joined.columns if c not in {sample_id_column, study_column, condition_column}]
    x_np = joined[feature_cols].astype(float).to_numpy(dtype=np.float32)
    x_std_np, _mean, _std = _standardize(x_np)

    study_values = sorted(joined[study_column].astype(str).unique())
    condition_values = sorted(joined[condition_column].astype(str).unique())
    study_map = {label: i for i, label in enumerate(study_values)}
    condition_map = {label: i for i, label in enumerate(condition_values)}
    y_study_np = np.array([study_map[v] for v in joined[study_column].astype(str)], dtype=np.int64)
    y_condition_np = np.array([condition_map[v] for v in joined[condition_column].astype(str)], dtype=np.int64)

    x = torch.tensor(x_std_np, dtype=torch.float32, device=torch_device)
    y_study = torch.tensor(y_study_np, dtype=torch.long, device=torch_device)
    y_condition = torch.tensor(y_condition_np, dtype=torch.long, device=torch_device)

    input_dim = int(x.shape[1])
    latent_dim = int(latent_dim or input_dim)
    encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.LayerNorm(latent_dim)).to(torch_device)
    simple_decoder = None
    study_decoder = None
    if use_study_conditioned_decoder:
        study_decoder = StudyConditionedDecoder(latent_dim, len(study_map), input_dim, hidden_dim=hidden_dim).to(torch_device)
    elif latent_dim != input_dim:
        simple_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        ).to(torch_device)

    condition_embedding = None
    study_adv_input_dim = latent_dim
    if condition_aware_adversary:
        condition_embedding = nn.Embedding(len(condition_map), condition_embedding_dim).to(torch_device)
        study_adv_input_dim += condition_embedding_dim

    study_adv = MLPAdversary(study_adv_input_dim, len(study_map), hidden_dim=hidden_dim).to(torch_device)
    condition_head = MLPAdversary(latent_dim, len(condition_map), hidden_dim=hidden_dim).to(torch_device)

    modules = [encoder, study_adv, condition_head]
    if simple_decoder is not None:
        modules.append(simple_decoder)
    if study_decoder is not None:
        modules.append(study_decoder)
    if condition_embedding is not None:
        modules.append(condition_embedding)
    optimizer = torch.optim.Adam([p for module in modules for p in module.parameters()], lr=lr)

    study_weights = _class_weights(y_study_np, len(study_map), torch_device, use_class_weights)
    condition_weights = _class_weights(y_condition_np, len(condition_map), torch_device, use_class_weights)

    dataset = TensorDataset(x, y_study, y_condition)
    if full_batch or batch_size is None or batch_size <= 0:
        effective_batch_size = len(dataset)
    else:
        effective_batch_size = min(int(batch_size), len(dataset))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )

    history: list[dict[str, Any]] = []
    sample_ids = joined[sample_id_column].astype(str).tolist()
    for epoch in range(1, epochs + 1):
        encoder.train()
        study_adv.train()
        condition_head.train()
        if simple_decoder is not None:
            simple_decoder.train()
        if study_decoder is not None:
            study_decoder.train()

        epoch_rows = []
        current_lambda = grl_lambda_for_epoch(epoch, epochs, lambda_grl, lambda_schedule, warmup_fraction)
        for batch_x, batch_study, batch_condition in loader:
            optimizer.zero_grad(set_to_none=True)
            z = encoder(batch_x)
            z_for_adv = gradient_reverse(z, current_lambda)
            if condition_embedding is not None:
                cond_context = condition_embedding(batch_condition)
                adv_input = torch.cat([z_for_adv, cond_context], dim=1)
            else:
                adv_input = z_for_adv
            study_logits = study_adv(adv_input)
            condition_logits = condition_head(z)
            study_loss = F.cross_entropy(study_logits, batch_study, weight=study_weights)
            condition_loss = F.cross_entropy(condition_logits, batch_condition, weight=condition_weights)
            if study_decoder is not None:
                recon = study_decoder(z, batch_study)
                preservation_loss = F.mse_loss(recon, batch_x)
            elif simple_decoder is not None:
                recon = simple_decoder(z)
                preservation_loss = F.mse_loss(recon, batch_x)
            else:
                preservation_loss = F.mse_loss(z, batch_x)
            loss = study_loss + condition_weight * condition_loss + preserve_weight * preservation_loss
            loss.backward()
            optimizer.step()
            epoch_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "study_adversary_loss": float(study_loss.detach().cpu()),
                    "condition_loss": float(condition_loss.detach().cpu()),
                    "preservation_loss": float(preservation_loss.detach().cpu()),
                    "n_batch_samples": int(batch_x.shape[0]),
                }
            )

        epoch_summary = {
            "epoch": epoch,
            "lambda_grl": current_lambda,
            "loss": float(np.average([r["loss"] for r in epoch_rows], weights=[r["n_batch_samples"] for r in epoch_rows])),
            "study_adversary_loss": float(
                np.average([r["study_adversary_loss"] for r in epoch_rows], weights=[r["n_batch_samples"] for r in epoch_rows])
            ),
            "condition_loss": float(
                np.average([r["condition_loss"] for r in epoch_rows], weights=[r["n_batch_samples"] for r in epoch_rows])
            ),
            "preservation_loss": float(
                np.average([r["preservation_loss"] for r in epoch_rows], weights=[r["n_batch_samples"] for r in epoch_rows])
            ),
        }
        if external_eval_every and (epoch == epochs or epoch % int(external_eval_every) == 0):
            corrected = _encode_all(encoder, x, sample_ids, sample_id_column)
            probes = evaluate_external_probes(
                corrected,
                metadata,
                sample_id_column=sample_id_column,
                study_column=study_column,
                condition_column=condition_column,
            )
            for _, row in probes.iterrows():
                epoch_summary[f"external_{row['metric']}"] = row.get("value")
        history.append(epoch_summary)

    corrected = _encode_all(encoder, x, sample_ids, sample_id_column)
    final_probes = evaluate_external_probes(
        corrected,
        metadata,
        sample_id_column=sample_id_column,
        study_column=study_column,
        condition_column=condition_column,
    )
    return GRLTrainingResult(
        corrected_embeddings=corrected,
        history=pd.DataFrame(history),
        final_probe_metrics=final_probes,
        study_labels=study_map,
        condition_labels=condition_map,
        config=asdict(config),
    )


def _encode_all(encoder, x, sample_ids: list[str], sample_id_column: str) -> pd.DataFrame:
    import torch

    encoder.eval()
    with torch.no_grad():
        z_np = encoder(x).detach().cpu().numpy()
    return _make_corrected_frame(sample_ids, z_np, sample_id_column)


def save_grl_result(result: GRLTrainingResult, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "corrected_embeddings": output_dir / "biogpt_grl_corrected_cls_389.csv",
        "training_history": output_dir / "grl_training_history.csv",
        "final_probe_metrics": output_dir / "grl_final_external_probe_metrics.csv",
        "label_maps": output_dir / "grl_label_maps.json",
        "config": output_dir / "grl_config.json",
    }
    result.corrected_embeddings.to_csv(paths["corrected_embeddings"], index=False)
    result.history.to_csv(paths["training_history"], index=False)
    result.final_probe_metrics.to_csv(paths["final_probe_metrics"], index=False)
    paths["label_maps"].write_text(
        json.dumps(
            {
                "study_labels": result.study_labels,
                "condition_labels": result.condition_labels,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["config"].write_text(json.dumps(result.config, indent=2), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}

