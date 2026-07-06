from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.grl_correction.grl import gradient_reverse


BENCHMARK_DIR = PROJECT / "crc_controlled_benchmark"
DATA_DIR = BENCHMARK_DIR / "data"
METHOD_DIR = BENCHMARK_DIR / "methods" / "scgpt_biomegpt"
REPORT_METHOD_DIR = BENCHMARK_DIR / "reports" / "methods"
METRIC_DIR = ROOT / "outputs" / "metrics"
REPORT_DIR = ROOT / "reports"
R_SCRIPT = PROJECT / "evaluate_crc_method.R"
RSCRIPT = Path(r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe")


@dataclass
class ResidualGRLConfig:
    method_name: str
    hidden_dim: int = 256
    epochs: int = 160
    lr: float = 1e-3
    lambda_grl: float = 0.2
    warmup_fraction: float = 0.2
    delta_scale: float = 0.5
    dropout: float = 0.05
    study_weight: float = 1.0
    abund_weight: float = 2.0
    sqrt_weight: float = 1.0
    cov_weight: float = 0.05
    delta_weight: float = 0.5
    anchor_weight: float = 0.1
    condition_context: bool = True
    seed: int = 42


CONFIGS = [
    ResidualGRLConfig(
        method_name="mmguide_resgrl_lam02_anchor01_pres2_cov005_delta05",
        lambda_grl=0.2,
        anchor_weight=0.1,
        abund_weight=2.0,
        cov_weight=0.05,
        delta_weight=0.5,
        delta_scale=0.5,
    ),
    ResidualGRLConfig(
        method_name="mmguide_resgrl_lam05_anchor01_pres2_cov005_delta05",
        lambda_grl=0.5,
        anchor_weight=0.1,
        abund_weight=2.0,
        cov_weight=0.05,
        delta_weight=0.5,
        delta_scale=0.5,
    ),
    ResidualGRLConfig(
        method_name="mmguide_resgrl_lam02_anchor02_pres5_cov01_delta1",
        lambda_grl=0.2,
        anchor_weight=0.2,
        abund_weight=5.0,
        cov_weight=0.1,
        delta_weight=1.0,
        delta_scale=0.35,
    ),
    ResidualGRLConfig(
        method_name="mmguide_resgrl_lam05_noanchor_pres5_cov01_delta1",
        lambda_grl=0.5,
        anchor_weight=0.0,
        abund_weight=5.0,
        cov_weight=0.1,
        delta_weight=1.0,
        delta_scale=0.35,
    ),
    ResidualGRLConfig(
        method_name="mmguide_resgrl_lam02_anchor1_pres2_cov005_delta05",
        lambda_grl=0.2,
        anchor_weight=1.0,
        abund_weight=2.0,
        cov_weight=0.05,
        delta_weight=0.5,
        delta_scale=0.5,
    ),
    ResidualGRLConfig(
        method_name="mmguide_resgrl_lam02_anchor2_pres1_cov005_delta1",
        lambda_grl=0.2,
        anchor_weight=2.0,
        abund_weight=1.0,
        cov_weight=0.05,
        delta_weight=0.5,
        delta_scale=0.75,
    ),
    ResidualGRLConfig(
        method_name="mmguide_resgrl_lam05_anchor1_pres1_cov005_delta1",
        lambda_grl=0.5,
        anchor_weight=1.0,
        abund_weight=1.0,
        cov_weight=0.05,
        delta_weight=0.5,
        delta_scale=0.75,
    ),
]


def read_abundance(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    df = pd.read_csv(path)
    features = df["feature"].astype(str).tolist()
    sample_ids = [c for c in df.columns if c != "feature"]
    abundance = df[sample_ids].astype(float).to_numpy()
    return features, sample_ids, abundance


def make_labels(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    studies = sorted(metadata["studyID"].astype(str).unique())
    conditions = sorted(metadata["study_condition"].astype(str).unique())
    study_map = {label: i for i, label in enumerate(studies)}
    condition_map = {label: i for i, label in enumerate(conditions)}
    y_study = metadata["studyID"].astype(str).map(study_map).to_numpy(dtype=np.int64)
    y_condition = metadata["study_condition"].astype(str).map(condition_map).to_numpy(dtype=np.int64)
    return y_study, y_condition, study_map, condition_map


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


def covariance_matrix(x):
    x = x - x.mean(dim=0, keepdim=True)
    return x.T @ x / max(1, x.shape[0] - 1)


def sqrt_relative_tensor(x):
    import torch

    positive = torch.clamp(x, min=0.0)
    relative = positive / torch.clamp(positive.sum(dim=1, keepdim=True), min=1e-12)
    return torch.sqrt(torch.clamp(relative, min=0.0) + 1e-12)


def effective_rank(x: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def pca_variance(x: np.ndarray) -> tuple[float, float]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    pca = PCA(n_components=2, random_state=0).fit(StandardScaler().fit_transform(x))
    return float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])


def train_one(config: ResidualGRLConfig) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cpu")

    features, sample_ids, raw_abundance = read_abundance(DATA_DIR / "crc_raw_abundance.csv")
    mm_features, mm_sample_ids, mmuphin_abundance = read_abundance(DATA_DIR / "crc_mmuphin_adjusted_abundance.csv")
    if features != mm_features or sample_ids != mm_sample_ids:
        raise ValueError("Raw and MMUPHin abundance matrices are not aligned.")
    metadata = pd.read_csv(DATA_DIR / "crc_metadata.csv", dtype=str)
    if sample_ids != metadata["sample_id"].astype(str).tolist():
        raise ValueError("Raw abundance sample order does not match metadata.")

    # Samples x features.
    raw_samples = np.clip(raw_abundance.T, 0.0, None).astype(np.float32)
    mmuphin_samples = np.clip(mmuphin_abundance.T, 0.0, None).astype(np.float32)
    raw_log = np.log1p(1000.0 * raw_samples).astype(np.float32)
    mm_log = np.log1p(1000.0 * mmuphin_samples).astype(np.float32)
    mean = raw_log.mean(axis=0, keepdims=True).astype(np.float32)
    std = raw_log.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    raw_std_np = ((raw_log - mean) / std).astype(np.float32)
    mm_std_np = ((mm_log - mean) / std).astype(np.float32)
    y_study_np, y_condition_np, study_map, condition_map = make_labels(metadata)

    x = torch.tensor(raw_std_np, dtype=torch.float32, device=device)
    mm_anchor = torch.tensor(mm_std_np, dtype=torch.float32, device=device)
    raw_abund = torch.tensor(raw_samples, dtype=torch.float32, device=device)
    y_study = torch.tensor(y_study_np, dtype=torch.long, device=device)
    y_condition = torch.tensor(y_condition_np, dtype=torch.long, device=device)
    condition_onehot = F.one_hot(y_condition, num_classes=len(condition_map)).float()

    input_dim = x.shape[1]
    adapter_in = input_dim + (len(condition_map) if config.condition_context else 0)
    adapter = nn.Sequential(
        nn.LayerNorm(adapter_in),
        nn.Linear(adapter_in, config.hidden_dim),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, input_dim),
    ).to(device)
    adversary = nn.Sequential(
        nn.Linear(input_dim + len(condition_map), config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, len(study_map)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(adversary.parameters()),
        lr=config.lr,
        weight_decay=1e-4,
    )
    study_weights = class_weights(y_study_np, len(study_map), device)
    cov_raw = covariance_matrix(x).detach()
    sqrt_raw = sqrt_relative_tensor(raw_abund).detach()
    raw_sample_sums = torch.clamp(raw_abund.sum(dim=1, keepdim=True), min=1e-12)

    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        adapter.train()
        adversary.train()
        optimizer.zero_grad(set_to_none=True)
        adapter_input = torch.cat([x, condition_onehot], dim=1) if config.condition_context else x
        delta = config.delta_scale * torch.tanh(adapter(adapter_input))
        corrected_std = x + delta
        corrected_log = corrected_std * torch.tensor(std, dtype=torch.float32, device=device) + torch.tensor(
            mean, dtype=torch.float32, device=device
        )
        corrected_abund = torch.clamp(torch.expm1(corrected_log) / 1000.0, min=0.0)
        corrected_abund = corrected_abund * (raw_sample_sums / torch.clamp(corrected_abund.sum(dim=1, keepdim=True), min=1e-12))

        lam = lambda_for_epoch(epoch, config.epochs, config.lambda_grl, config.warmup_fraction)
        adv_input = torch.cat([gradient_reverse(corrected_std, lam), condition_onehot.detach()], dim=1)
        study_loss = F.cross_entropy(adversary(adv_input), y_study, weight=study_weights)
        abund_loss = F.mse_loss(corrected_std, x)
        sqrt_loss = F.mse_loss(sqrt_relative_tensor(corrected_abund), sqrt_raw)
        cov_loss = F.mse_loss(covariance_matrix(corrected_std), cov_raw)
        delta_loss = torch.mean(delta**2)
        anchor_loss = F.mse_loss(corrected_std, mm_anchor)
        loss = (
            config.study_weight * study_loss
            + config.abund_weight * abund_loss
            + config.sqrt_weight * sqrt_loss
            + config.cov_weight * cov_loss
            + config.delta_weight * delta_loss
            + config.anchor_weight * anchor_loss
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            history.append(
                {
                    "method": config.method_name,
                    "epoch": epoch,
                    "lambda_grl": lam,
                    "loss": float(loss.detach().cpu()),
                    "study_loss": float(study_loss.detach().cpu()),
                    "abund_loss": float(abund_loss.detach().cpu()),
                    "sqrt_loss": float(sqrt_loss.detach().cpu()),
                    "cov_loss": float(cov_loss.detach().cpu()),
                    "delta_loss": float(delta_loss.detach().cpu()),
                    "anchor_loss": float(anchor_loss.detach().cpu()),
                }
            )

    adapter.eval()
    with torch.no_grad():
        adapter_input = torch.cat([x, condition_onehot], dim=1) if config.condition_context else x
        delta = config.delta_scale * torch.tanh(adapter(adapter_input))
        corrected_std = x + delta
        corrected_log = corrected_std * torch.tensor(std, dtype=torch.float32, device=device) + torch.tensor(
            mean, dtype=torch.float32, device=device
        )
        corrected_abund = torch.clamp(torch.expm1(corrected_log) / 1000.0, min=0.0)
        corrected_abund = corrected_abund * (raw_sample_sums / torch.clamp(corrected_abund.sum(dim=1, keepdim=True), min=1e-12))
        corrected_samples = corrected_abund.detach().cpu().numpy()
        corrected_std_np = corrected_std.detach().cpu().numpy()
        delta_np = delta.detach().cpu().numpy()

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = METHOD_DIR / f"{config.method_name}.csv"
    out_df = pd.DataFrame(corrected_samples.T, columns=sample_ids)
    out_df.insert(0, "feature", features)
    out_df.to_csv(out_csv, index=False)
    pc1, pc2 = pca_variance(corrected_std_np)
    return {
        "method_name": config.method_name,
        "matrix_path": str(out_csv),
        "config": asdict(config),
        "history": history,
        "effective_rank": effective_rank(corrected_std_np),
        "pc1_variance": pc1,
        "pc2_variance": pc2,
        "mean_delta_l2": float(np.mean(np.linalg.norm(delta_np, axis=1))),
        "mean_delta_mse": float(np.mean(delta_np**2)),
    }


def run_r_evaluator(method_name: str, matrix_path: str) -> None:
    command = [
        str(RSCRIPT if RSCRIPT.exists() else "Rscript"),
        str(R_SCRIPT),
        "--method",
        method_name,
        "--matrix",
        matrix_path,
    ]
    subprocess.run(command, cwd=PROJECT, check=True)


def collect_primary_metrics(method_names: list[str]) -> pd.DataFrame:
    rows = []
    for method in method_names:
        path = REPORT_METHOD_DIR / method / f"{method}_primary_comparison.csv"
        if path.exists():
            df = pd.read_csv(path)
            rows.append(df[df["method"] == method])
    baseline = pd.read_csv(BENCHMARK_DIR / "reports" / "crc_raw_vs_mmuphin_metrics.csv")
    rows.append(baseline[baseline["method"].isin(["raw", "mmuphin"])])
    long = pd.concat(rows, ignore_index=True)
    primary = long[long["metric"].isin(
        [
            "study_R2_condition_controlled",
            "study_prediction_balanced_accuracy",
            "disease_LOSO_mean_within_study_AUC",
            "condition_R2_study_controlled",
            "disease_LOSO_balanced_accuracy",
        ]
    )].copy()
    wide = primary.pivot_table(index="method", columns="metric", values="estimate", aggfunc="first").reset_index()
    order = {"raw": 0, "mmuphin": 1}
    order.update({name: i + 2 for i, name in enumerate(method_names)})
    wide["_order"] = wide["method"].map(order)
    return wide.sort_values("_order").drop(columns="_order")


def write_summary(comparison: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MMUPHin-Guided Residual GRL Full551",
        "",
        "This is a rescue experiment inspired by MMUPHin: instead of compressing abundance to an 8D latent and freely reconstructing, the model learns a small feature-space residual correction.",
        "",
        "```text",
        "x_corrected = x_raw + delta",
        "```",
        "",
        "Losses include conditional study GRL, abundance preservation, sqrt-relative-abundance preservation, covariance preservation, delta penalty, and optional MMUPHin anchor.",
        "",
        "## R Evaluator Primary Metrics",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Geometry Diagnostics",
        "",
        diagnostics.to_markdown(index=False),
        "",
        "## Reading Rules",
        "",
        "- Success requires lower study signal than MMUPHin while preserving disease LOSO AUC and avoiding low-rank geometry.",
        "- Full-data results remain diagnostic only until strict LOSO/cross-fitted correction is run.",
        "- If effective rank is far below raw/MMUPHin or PC1/PC2 variance is inflated, treat the result as artificial geometry.",
        "",
    ]
    (REPORT_DIR / "mmuphin_guided_residual_grl_full551_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics_rows = []
    history_rows = []
    method_names = []
    configs = []
    for config in CONFIGS:
        print("training", config.method_name)
        result = train_one(config)
        method_names.append(config.method_name)
        configs.append({k: v for k, v in result.items() if k not in {"history"}})
        history_rows.extend(result["history"])
        diagnostics_rows.append(
            {
                "method": config.method_name,
                "effective_rank": result["effective_rank"],
                "pc1_variance": result["pc1_variance"],
                "pc2_variance": result["pc2_variance"],
                "mean_delta_l2": result["mean_delta_l2"],
                "mean_delta_mse": result["mean_delta_mse"],
            }
        )
        print("evaluating", config.method_name)
        run_r_evaluator(config.method_name, result["matrix_path"])

    comparison = collect_primary_metrics(method_names)
    diagnostics = pd.DataFrame(diagnostics_rows)
    comparison.to_csv(METRIC_DIR / "mmuphin_guided_residual_grl_full551_comparison.csv", index=False)
    diagnostics.to_csv(METRIC_DIR / "mmuphin_guided_residual_grl_full551_diagnostics.csv", index=False)
    pd.DataFrame(history_rows).to_csv(METRIC_DIR / "mmuphin_guided_residual_grl_full551_training_history.csv", index=False)
    (METRIC_DIR / "mmuphin_guided_residual_grl_full551_configs.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")
    write_summary(comparison, diagnostics)
    print("MMUPHIN_GUIDED_RESIDUAL_GRL_FULL551_OK")
    print(comparison.to_string(index=False))
    print(diagnostics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
