from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.grl_correction.grl import gradient_reverse


BENCHMARK_DIR = PROJECT_ROOT / "crc_controlled_benchmark"
DATA_DIR = BENCHMARK_DIR / "data"
METHOD_DIR = BENCHMARK_DIR / "methods" / "scgpt_biomegpt"
OUTPUT_DIR = ROOT / "outputs" / "crc_full_abundance_grl"
METRIC_DIR = ROOT / "outputs" / "metrics"


@dataclass
class AbundanceGRLConfig:
    method_name: str
    latent_dim: int = 8
    hidden_dim: int = 128
    epochs: int = 200
    lr: float = 1e-3
    lambda_grl: float = 10.0
    condition_weight: float = 0.1
    recon_weight: float = 1.0
    rel_recon_weight: float = 0.0
    variance_weight: float = 0.0
    use_condition_head: bool = True
    condition_context_for_adversary: bool = True
    study_conditioned_decoder: bool = False
    study_embedding_dim: int = 16
    dropout: float = 0.1
    batch_size: int = 64
    seed: int = 42
    warmup_fraction: float = 0.1
    renormalize_to_raw_sample_sum: bool = True


CONFIGS = [
    AbundanceGRLConfig(
        method_name="grl_abundance_l8_lam10_cw01_rw1",
        latent_dim=8,
        lambda_grl=10.0,
        condition_weight=0.1,
        recon_weight=1.0,
    ),
    AbundanceGRLConfig(
        method_name="grl_abundance_l16_lam10_cw01_rw1",
        latent_dim=16,
        lambda_grl=10.0,
        condition_weight=0.1,
        recon_weight=1.0,
    ),
    AbundanceGRLConfig(
        method_name="grl_abundance_l8_lam5_cw05_rw1",
        latent_dim=8,
        lambda_grl=5.0,
        condition_weight=0.5,
        recon_weight=1.0,
    ),
    AbundanceGRLConfig(
        method_name="grl_abundance_l8_lam10_cw001_rw1",
        latent_dim=8,
        lambda_grl=10.0,
        condition_weight=0.01,
        recon_weight=1.0,
    ),
    AbundanceGRLConfig(
        method_name="grl_abundance_l8_lam10_cw0_rw1",
        latent_dim=8,
        lambda_grl=10.0,
        condition_weight=0.0,
        recon_weight=1.0,
    ),
    AbundanceGRLConfig(
        method_name="grl_mech_context_only_l8_lam10_rw5_rel1_var1",
        latent_dim=8,
        hidden_dim=128,
        lambda_grl=10.0,
        condition_weight=0.0,
        recon_weight=5.0,
        rel_recon_weight=1.0,
        variance_weight=1.0,
        use_condition_head=False,
        condition_context_for_adversary=True,
        study_conditioned_decoder=False,
    ),
    AbundanceGRLConfig(
        method_name="grl_mech_studydec_context_l8_lam10_rw5_rel1_var1",
        latent_dim=8,
        hidden_dim=128,
        lambda_grl=10.0,
        condition_weight=0.0,
        recon_weight=5.0,
        rel_recon_weight=1.0,
        variance_weight=1.0,
        use_condition_head=False,
        condition_context_for_adversary=True,
        study_conditioned_decoder=True,
    ),
    AbundanceGRLConfig(
        method_name="grl_mech_context_only_l16_lam10_rw5_rel1_var1",
        latent_dim=16,
        hidden_dim=128,
        lambda_grl=10.0,
        condition_weight=0.0,
        recon_weight=5.0,
        rel_recon_weight=1.0,
        variance_weight=1.0,
        use_condition_head=False,
        condition_context_for_adversary=True,
        study_conditioned_decoder=False,
    ),
]


def read_abundance(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    df = pd.read_csv(path)
    features = df["feature"].astype(str).tolist()
    sample_ids = [c for c in df.columns if c != "feature"]
    abundance = df[sample_ids].astype(float).to_numpy()
    return features, sample_ids, abundance


def class_weights(labels: np.ndarray, n_classes: int, device: Any):
    import torch

    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def lambda_for_epoch(epoch: int, epochs: int, lambda_grl: float, warmup_fraction: float) -> float:
    progress = (epoch - 1) / max(epochs - 1, 1)
    warmup = min(progress / max(warmup_fraction, 1e-8), 1.0)
    return float(lambda_grl * warmup)


def make_labels(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    study_values = sorted(metadata["studyID"].astype(str).unique())
    condition_values = sorted(metadata["study_condition"].astype(str).unique())
    study_map = {label: i for i, label in enumerate(study_values)}
    condition_map = {label: i for i, label in enumerate(condition_values)}
    y_study = np.array([study_map[v] for v in metadata["studyID"].astype(str)], dtype=np.int64)
    y_condition = np.array([condition_map[v] for v in metadata["study_condition"].astype(str)], dtype=np.int64)
    return y_study, y_condition, study_map, condition_map


def train_one(config: AbundanceGRLConfig) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cpu")

    features, sample_ids, abundance = read_abundance(DATA_DIR / "crc_raw_abundance.csv")
    metadata = pd.read_csv(DATA_DIR / "crc_metadata.csv", dtype=str)
    if sample_ids != metadata["sample_id"].astype(str).tolist():
        raise ValueError("Raw abundance sample order does not match metadata.")

    x_log = np.log1p(1000.0 * np.clip(abundance.T, 0, None)).astype(np.float32)
    mean = x_log.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_log.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    x_std = ((x_log - mean) / std).astype(np.float32)
    y_study_np, y_condition_np, study_map, condition_map = make_labels(metadata)

    x = torch.tensor(x_std, dtype=torch.float32, device=device)
    y_study = torch.tensor(y_study_np, dtype=torch.long, device=device)
    y_condition = torch.tensor(y_condition_np, dtype=torch.long, device=device)
    input_dim = x.shape[1]
    latent_dim = min(config.latent_dim, input_dim)

    encoder = nn.Sequential(
        nn.Linear(input_dim, config.hidden_dim),
        nn.ReLU(),
        nn.Linear(config.hidden_dim, latent_dim),
        nn.LayerNorm(latent_dim),
    ).to(device)
    decoder = nn.Sequential(
        nn.Linear(latent_dim + (config.study_embedding_dim if config.study_conditioned_decoder else 0), config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, input_dim),
    ).to(device)
    condition_embedding = nn.Embedding(len(condition_map), 16).to(device) if config.condition_context_for_adversary else None
    study_embedding = nn.Embedding(len(study_map), config.study_embedding_dim).to(device) if config.study_conditioned_decoder else None
    study_adversary = nn.Sequential(
        nn.Linear(latent_dim + (16 if config.condition_context_for_adversary else 0), config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, len(study_map)),
    ).to(device)
    condition_head = nn.Sequential(
        nn.Linear(latent_dim, config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, len(condition_map)),
    ).to(device) if config.use_condition_head else None
    modules = [encoder, decoder, study_adversary]
    if condition_embedding is not None:
        modules.append(condition_embedding)
    if study_embedding is not None:
        modules.append(study_embedding)
    if condition_head is not None:
        modules.append(condition_head)
    optimizer = torch.optim.Adam([p for module in modules for p in module.parameters()], lr=config.lr)
    study_weights = class_weights(y_study_np, len(study_map), device)
    condition_weights = class_weights(y_condition_np, len(condition_map), device)

    loader = DataLoader(
        TensorDataset(x, y_study, y_condition),
        batch_size=min(config.batch_size, len(x)),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(config.seed),
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        batch_rows = []
        current_lambda = lambda_for_epoch(epoch, config.epochs, config.lambda_grl, config.warmup_fraction)
        encoder.train()
        decoder.train()
        study_adversary.train()
        if condition_head is not None:
            condition_head.train()
        if condition_embedding is not None:
            condition_embedding.train()
        if study_embedding is not None:
            study_embedding.train()
        for batch_x, batch_study, batch_condition in loader:
            optimizer.zero_grad(set_to_none=True)
            z = encoder(batch_x)
            decoder_input = torch.cat([z, study_embedding(batch_study)], dim=1) if study_embedding is not None else z
            recon = decoder(decoder_input)
            z_adv = gradient_reverse(z, current_lambda)
            adv_input = torch.cat([z_adv, condition_embedding(batch_condition)], dim=1) if condition_embedding is not None else z_adv
            study_loss = F.cross_entropy(study_adversary(adv_input), batch_study, weight=study_weights)
            if condition_head is not None:
                condition_loss = F.cross_entropy(condition_head(z), batch_condition, weight=condition_weights)
            else:
                condition_loss = torch.zeros((), dtype=torch.float32, device=device)
            recon_loss = F.mse_loss(recon, batch_x)
            rel_recon_loss = torch.zeros((), dtype=torch.float32, device=device)
            if config.rel_recon_weight > 0:
                rel_recon_loss = F.mse_loss(torch.softmax(recon, dim=1), torch.softmax(batch_x, dim=1))
            variance_loss = torch.zeros((), dtype=torch.float32, device=device)
            if config.variance_weight > 0 and batch_x.shape[0] > 1:
                variance_loss = F.mse_loss(recon.var(dim=0, unbiased=False), batch_x.var(dim=0, unbiased=False))
            loss = (
                study_loss
                + config.condition_weight * condition_loss
                + config.recon_weight * recon_loss
                + config.rel_recon_weight * rel_recon_loss
                + config.variance_weight * variance_loss
            )
            loss.backward()
            optimizer.step()
            batch_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "study_loss": float(study_loss.detach().cpu()),
                    "condition_loss": float(condition_loss.detach().cpu()),
                    "recon_loss": float(recon_loss.detach().cpu()),
                    "rel_recon_loss": float(rel_recon_loss.detach().cpu()),
                    "variance_loss": float(variance_loss.detach().cpu()),
                    "n": int(batch_x.shape[0]),
                }
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            history.append(
                {
                    "method": config.method_name,
                    "epoch": epoch,
                    "lambda_grl": current_lambda,
                    "loss": float(np.average([r["loss"] for r in batch_rows], weights=[r["n"] for r in batch_rows])),
                    "study_loss": float(np.average([r["study_loss"] for r in batch_rows], weights=[r["n"] for r in batch_rows])),
                    "condition_loss": float(np.average([r["condition_loss"] for r in batch_rows], weights=[r["n"] for r in batch_rows])),
                    "recon_loss": float(np.average([r["recon_loss"] for r in batch_rows], weights=[r["n"] for r in batch_rows])),
                    "rel_recon_loss": float(np.average([r["rel_recon_loss"] for r in batch_rows], weights=[r["n"] for r in batch_rows])),
                    "variance_loss": float(np.average([r["variance_loss"] for r in batch_rows], weights=[r["n"] for r in batch_rows])),
                }
            )

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z_all = encoder(x)
        decoder_input = torch.cat([z_all, study_embedding(y_study)], dim=1) if study_embedding is not None else z_all
        recon_std = decoder(decoder_input).detach().cpu().numpy()
    recon_log = recon_std * std + mean
    recon_abundance_samples = np.expm1(recon_log) / 1000.0
    recon_abundance_samples = np.clip(recon_abundance_samples, 0.0, None)
    if config.renormalize_to_raw_sample_sum:
        raw_sample_sums = np.clip(abundance.sum(axis=0, keepdims=True).T, 1e-12, None)
        recon_sums = np.clip(recon_abundance_samples.sum(axis=1, keepdims=True), 1e-12, None)
        recon_abundance_samples = recon_abundance_samples * (raw_sample_sums / recon_sums)
    recon_abundance = recon_abundance_samples.T

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = METHOD_DIR / f"{config.method_name}.csv"
    mirror_csv = OUTPUT_DIR / f"{config.method_name}.csv"
    out_df = pd.DataFrame(recon_abundance, columns=sample_ids)
    out_df.insert(0, "feature", features)
    out_df.to_csv(out_csv, index=False)
    out_df.to_csv(mirror_csv, index=False)
    return {
        "method_name": config.method_name,
        "matrix_path": str(out_csv),
        "mirror_matrix_path": str(mirror_csv),
        "final_recon_loss": history[-1]["recon_loss"],
        "final_study_loss": history[-1]["study_loss"],
        "final_condition_loss": history[-1]["condition_loss"],
        "config": asdict(config),
        "history": history,
    }


def main() -> int:
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    all_history = []
    for config in CONFIGS:
        result = train_one(config)
        results.append({k: v for k, v in result.items() if k != "history"})
        all_history.extend(result["history"])
        print("trained", config.method_name, "recon_loss=", result["final_recon_loss"])
    summary_path = METRIC_DIR / "grl_abundance_decoder_full_crc_training_summary.json"
    history_path = METRIC_DIR / "grl_abundance_decoder_full_crc_training_history.csv"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(all_history).to_csv(history_path, index=False)
    print("GRL_ABUNDANCE_DECODER_FULL_CRC_OK")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
