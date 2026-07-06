from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_biogpt_cls_crc389 import study_mean_center_cls
from scripts.evaluate_mmuphin_style_crc389 import evaluate_method
from src.evaluation.io import read_matrix
from src.evaluation.plots import save_pca_plot
from src.grl_correction.grl import gradient_reverse


DATA_DIR = ROOT / "outputs" / "crc_overlap_benchmark"
METRIC_DIR = ROOT / "outputs" / "metrics"
FIGURE_DIR = ROOT / "outputs" / "figures" / "mechanism_grl_crc389"
REPORT_DIR = ROOT / "reports"


def align_matrix(matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    ids = metadata["sample_id"].astype(str).tolist()
    matrix = matrix.copy()
    matrix["sample_id"] = matrix["sample_id"].astype(str)
    return matrix.set_index("sample_id").loc[ids].reset_index()


def label_maps(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    study_values = sorted(metadata["studyID"].astype(str).unique())
    condition_values = sorted(metadata["study_condition"].astype(str).unique())
    study_map = {label: i for i, label in enumerate(study_values)}
    condition_map = {label: i for i, label in enumerate(condition_values)}
    y_study = np.array([study_map[v] for v in metadata["studyID"].astype(str)], dtype=np.int64)
    y_condition = np.array([condition_map[v] for v in metadata["study_condition"].astype(str)], dtype=np.int64)
    return y_study, y_condition, study_map, condition_map


def class_weights(y: np.ndarray, n_classes: int, device: Any):
    import torch

    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def lambda_for_epoch(epoch: int, epochs: int, lambda_grl: float, warmup_fraction: float) -> float:
    progress = (epoch - 1) / max(epochs - 1, 1)
    return float(lambda_grl * min(progress / max(warmup_fraction, 1e-8), 1.0))


def train_mechanism_grl_389(raw: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    config = {
        "method": "mechanism_grl_abundance_389",
        "latent_dim": 8,
        "hidden_dim": 128,
        "epochs": 200,
        "lr": 1e-3,
        "lambda_grl": 10.0,
        "warmup_fraction": 0.1,
        "recon_weight": 5.0,
        "rel_recon_weight": 1.0,
        "variance_weight": 1.0,
        "condition_head": False,
        "condition_context_for_adversary": True,
        "batch_size": 64,
        "seed": 42,
    }
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = torch.device("cpu")
    raw = align_matrix(raw, metadata)
    feature_cols = [c for c in raw.columns if c != "sample_id"]
    x_raw = raw[feature_cols].astype(float).to_numpy()
    x_log = np.log1p(1000.0 * np.clip(x_raw, 0, None)).astype(np.float32)
    mean = x_log.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_log.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    x_std = ((x_log - mean) / std).astype(np.float32)
    y_study_np, y_condition_np, study_map, condition_map = label_maps(metadata)

    x = torch.tensor(x_std, dtype=torch.float32, device=device)
    y_study = torch.tensor(y_study_np, dtype=torch.long, device=device)
    y_condition = torch.tensor(y_condition_np, dtype=torch.long, device=device)
    input_dim = x.shape[1]
    latent_dim = min(config["latent_dim"], input_dim)
    encoder = nn.Sequential(
        nn.Linear(input_dim, config["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(config["hidden_dim"], latent_dim),
        nn.LayerNorm(latent_dim),
    ).to(device)
    decoder = nn.Sequential(
        nn.Linear(latent_dim, config["hidden_dim"]),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(config["hidden_dim"], input_dim),
    ).to(device)
    condition_embedding = nn.Embedding(len(condition_map), 16).to(device)
    study_adversary = nn.Sequential(
        nn.Linear(latent_dim + 16, config["hidden_dim"]),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(config["hidden_dim"], len(study_map)),
    ).to(device)
    modules = [encoder, decoder, condition_embedding, study_adversary]
    optimizer = torch.optim.Adam([p for module in modules for p in module.parameters()], lr=config["lr"])
    study_weights = class_weights(y_study_np, len(study_map), device)
    loader = DataLoader(
        TensorDataset(x, y_study, y_condition),
        batch_size=min(config["batch_size"], len(x)),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(config["seed"]),
    )
    history = []
    for epoch in range(1, config["epochs"] + 1):
        current_lambda = lambda_for_epoch(epoch, config["epochs"], config["lambda_grl"], config["warmup_fraction"])
        batch_rows = []
        for batch_x, batch_study, batch_condition in loader:
            optimizer.zero_grad(set_to_none=True)
            z = encoder(batch_x)
            recon = decoder(z)
            adv_input = torch.cat([gradient_reverse(z, current_lambda), condition_embedding(batch_condition)], dim=1)
            study_loss = F.cross_entropy(study_adversary(adv_input), batch_study, weight=study_weights)
            recon_loss = F.mse_loss(recon, batch_x)
            rel_recon_loss = F.mse_loss(torch.softmax(recon, dim=1), torch.softmax(batch_x, dim=1))
            variance_loss = F.mse_loss(recon.var(dim=0, unbiased=False), batch_x.var(dim=0, unbiased=False))
            loss = (
                study_loss
                + config["recon_weight"] * recon_loss
                + config["rel_recon_weight"] * rel_recon_loss
                + config["variance_weight"] * variance_loss
            )
            loss.backward()
            optimizer.step()
            batch_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "study_loss": float(study_loss.detach().cpu()),
                    "recon_loss": float(recon_loss.detach().cpu()),
                    "rel_recon_loss": float(rel_recon_loss.detach().cpu()),
                    "variance_loss": float(variance_loss.detach().cpu()),
                    "n": int(batch_x.shape[0]),
                }
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == config["epochs"]:
            history.append(
                {
                    "epoch": epoch,
                    "lambda_grl": current_lambda,
                    **{
                        k: float(np.average([r[k] for r in batch_rows], weights=[r["n"] for r in batch_rows]))
                        for k in ["loss", "study_loss", "recon_loss", "rel_recon_loss", "variance_loss"]
                    },
                }
            )
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        recon_std = decoder(encoder(x)).detach().cpu().numpy()
    recon_log = recon_std * std + mean
    recon = np.expm1(recon_log) / 1000.0
    recon = np.clip(recon, 0, None)
    raw_sums = np.clip(x_raw.sum(axis=1, keepdims=True), 1e-12, None)
    recon_sums = np.clip(recon.sum(axis=1, keepdims=True), 1e-12, None)
    recon = recon * (raw_sums / recon_sums)
    out = pd.DataFrame(recon, columns=feature_cols)
    out.insert(0, "sample_id", raw["sample_id"].astype(str).to_numpy())
    return out, pd.DataFrame(history), config


def pivot_order(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics.pivot(index="method", columns="metric", values="estimate").reset_index()
    order = {
        "Raw abundance": 0,
        "MMUPHin adjusted abundance": 1,
        "Mechanism-only GRL abundance": 2,
        "BiomeGPT raw CLS": 3,
        "BiomeGPT study-mean-centered CLS": 4,
    }
    primary["_order"] = primary["method"].map(order)
    return primary.sort_values("_order").drop(columns="_order")


def main() -> int:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(DATA_DIR / "metadata_389.csv", dtype=str)
    raw = read_matrix(DATA_DIR / "raw_abundance_389.csv")
    grl, history, config = train_mechanism_grl_389(raw, metadata)
    grl_path = DATA_DIR / "mechanism_grl_abundance_389.csv"
    grl.to_csv(grl_path, index=False)
    history_path = METRIC_DIR / "mechanism_grl_crc389_training_history.csv"
    config_path = METRIC_DIR / "mechanism_grl_crc389_config.json"
    history.to_csv(history_path, index=False)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    raw_cls = read_matrix(DATA_DIR / "biogpt_raw_cls_389.csv")
    centered_cls_path = DATA_DIR / "biogpt_mean_centered_cls_389.csv"
    if centered_cls_path.exists():
        centered_cls = read_matrix(centered_cls_path)
    else:
        centered_cls = study_mean_center_cls(raw_cls, metadata)
        centered_cls.to_csv(centered_cls_path, index=False)

    specs = [
        ("Raw abundance", raw, True, "raw_abundance"),
        ("MMUPHin adjusted abundance", read_matrix(DATA_DIR / "mmuphin_adjusted_abundance_389.csv"), True, "mmuphin_adjusted_abundance"),
        ("Mechanism-only GRL abundance", grl, True, "mechanism_grl_abundance"),
        ("BiomeGPT raw CLS", raw_cls, False, "biogpt_raw_cls"),
        ("BiomeGPT study-mean-centered CLS", centered_cls, False, "biogpt_mean_centered_cls"),
    ]
    rows: list[dict[str, Any]] = []
    for method, matrix, abundance_like, slug in specs:
        matrix = align_matrix(matrix, metadata)
        rows.extend(evaluate_method(method, matrix, metadata, abundance_like=abundance_like))
        save_pca_plot(matrix, metadata, "studyID", FIGURE_DIR / f"{slug}_pca_by_study.png")
        save_pca_plot(matrix, metadata, "study_condition", FIGURE_DIR / f"{slug}_pca_by_condition.png")
        print("evaluated", method)
    metrics = pd.DataFrame(rows)
    long_path = METRIC_DIR / "mechanism_grl_crc389_metrics_long.csv"
    primary_path = METRIC_DIR / "mechanism_grl_crc389_primary_table.csv"
    metrics.to_csv(long_path, index=False)
    primary = pivot_order(metrics)
    primary.to_csv(primary_path, index=False)

    grl_row = primary[primary["method"] == "Mechanism-only GRL abundance"].iloc[0]
    mm_row = primary[primary["method"] == "MMUPHin adjusted abundance"].iloc[0]
    lines = [
        "# Mechanism-Only GRL on CRC389 Overlap",
        "",
        "## Scope",
        "",
        "This trains the mechanism-only abundance decoder on the 389 CRC overlap samples so the abundance result can be compared with BiomeGPT CLS baselines on the same sample set.",
        "",
        "This is a CRC389 diagnostic table, not the original full 551-sample MMUPHin R benchmark.",
        "",
        "## Primary Table",
        "",
        primary.to_markdown(index=False),
        "",
        "## Reading",
        "",
        f"- Mechanism-only GRL abundance Study BA is {grl_row['study_prediction_balanced_accuracy']:.3f} versus MMUPHin {mm_row['study_prediction_balanced_accuracy']:.3f}.",
        f"- Mechanism-only GRL abundance disease LOSO AUC is {grl_row['disease_LOSO_mean_within_study_AUC']:.3f} versus MMUPHin {mm_row['disease_LOSO_mean_within_study_AUC']:.3f}.",
        "- This table is mainly for same-sample linkage to BiomeGPT CLS; final MMUPHin-method claims should still use the full 551-sample R benchmark.",
        "",
        "## Output Files",
        "",
        f"- `mechanism_grl_abundance`: `{grl_path.relative_to(ROOT)}`",
        f"- `primary_table`: `{primary_path.relative_to(ROOT)}`",
        f"- `metrics_long`: `{long_path.relative_to(ROOT)}`",
        f"- `training_history`: `{history_path.relative_to(ROOT)}`",
        f"- `figures`: `{FIGURE_DIR.relative_to(ROOT)}`",
    ]
    (REPORT_DIR / "mechanism_grl_crc389_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("MECHANISM_GRL_CRC389_OK")
    print(primary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
