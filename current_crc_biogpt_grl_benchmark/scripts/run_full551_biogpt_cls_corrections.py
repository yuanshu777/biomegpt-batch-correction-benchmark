from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_biogpt_cls_crc389 import study_mean_center_cls
from scripts.evaluate_mmuphin_style_crc389 import evaluate_method, pivot_primary
from scripts.run_cls_architecture_ablation_crc389 import study_subspace_projection
from src.evaluation.io import read_matrix
from src.evaluation.plots import save_pca_plot
from src.grl_correction.nomean_adapter import NoMeanAdapterConfig, train_nomean_cls_adapter
from src.grl_correction.split_adapter import SplitAdapterConfig, train_split_cls_adapter


FULL_DATA = PROJECT / "crc_controlled_benchmark" / "data"
OUT_FULL = ROOT / "outputs" / "crc_full551_benchmark"
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_FIGURES = ROOT / "outputs" / "figures" / "full551_biogpt_cls"
REPORTS = ROOT / "reports"


def align_matrix(matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    ids = metadata["sample_id"].astype(str).tolist()
    out = matrix.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    return out.set_index("sample_id").loc[ids].reset_index()


def diagnostics(matrix: pd.DataFrame, metadata: pd.DataFrame, method: str) -> dict[str, Any]:
    import numpy as np
    from src.grl_correction.nomean_adapter import effective_rank, pc1_condition_auc

    x = matrix.drop(columns=["sample_id"]).astype(float).to_numpy()
    return {
        "method": method,
        "n_samples": x.shape[0],
        "n_dims": x.shape[1],
        "effective_rank": effective_rank(x),
        "pc1_condition_auc": pc1_condition_auc(x, metadata),
        "mean_abs_value": float(np.mean(np.abs(x))),
    }


def main() -> int:
    OUT_FULL.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(FULL_DATA / "crc_metadata.csv", dtype=str)
    raw_cls_path = OUT_FULL / "biogpt_raw_cls_551.csv"
    if not raw_cls_path.exists():
        raise FileNotFoundError(f"Missing full551 raw CLS: {raw_cls_path}")
    raw_cls = align_matrix(read_matrix(raw_cls_path), metadata)

    corrected_specs: list[tuple[str, Path, bool, str]] = [
        ("Raw abundance", FULL_DATA / "crc_raw_abundance.csv", True, "raw_abundance"),
        ("MMUPHin adjusted abundance", FULL_DATA / "crc_mmuphin_adjusted_abundance.csv", True, "mmuphin_adjusted_abundance"),
        ("BiomeGPT raw CLS 551", raw_cls_path, False, "biogpt_raw_cls_551"),
    ]
    diag_rows = [diagnostics(raw_cls, metadata, "BiomeGPT raw CLS 551")]
    histories: list[pd.DataFrame] = []
    configs: dict[str, Any] = {}

    centered = study_mean_center_cls(raw_cls, metadata)
    centered_path = OUT_FULL / "biogpt_study_mean_centered_cls_551.csv"
    centered.to_csv(centered_path, index=False)
    corrected_specs.append(("BiomeGPT study-mean-centered CLS 551", centered_path, False, "biogpt_mean_centered_cls_551"))
    diag_rows.append(diagnostics(centered, metadata, "BiomeGPT study-mean-centered CLS 551"))

    print("training full551 NoMean conditional GRL")
    nomean_config = NoMeanAdapterConfig(
        hidden_dim=128,
        study_embedding_dim=16,
        residual_scale=0.1,
        dropout=0.05,
        epochs=100,
        warmup_epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=1e-4,
        lambda_grl=0.5,
        lambda_schedule="linear",
        reconstruction_weight=1.0,
        preserve_weight=1.0,
        adversary_weight=0.1,
        adversary_mode="conditional",
        variance_weight=0.02,
        covariance_weight=0.005,
        seed=42,
    )
    nomean = train_nomean_cls_adapter(raw_cls, metadata, config=nomean_config)
    nomean_path = OUT_FULL / "biogpt_nomean_conditional_grl_cls_551.csv"
    nomean.corrected_embeddings.to_csv(nomean_path, index=False)
    corrected_specs.append(("BiomeGPT NoMean conditional GRL CLS 551", nomean_path, False, "biogpt_nomean_conditional_grl_cls_551"))
    diag_rows.append({"method": "BiomeGPT NoMean conditional GRL CLS 551", **nomean.diagnostics, "n_samples": len(metadata), "n_dims": 512})
    hist = nomean.training_history.copy()
    hist.insert(0, "method", "BiomeGPT NoMean conditional GRL CLS 551")
    histories.append(hist)
    configs["nomean_conditional_grl"] = nomean.config

    print("building full551 study-subspace projection")
    projection, projection_diag = study_subspace_projection(raw_cls, metadata)
    projection_path = OUT_FULL / "biogpt_study_subspace_projection_cls_551.csv"
    projection.to_csv(projection_path, index=False)
    corrected_specs.append(("BiomeGPT study-subspace projection CLS 551", projection_path, False, "biogpt_study_subspace_projection_cls_551"))
    diag_rows.append({"method": "BiomeGPT study-subspace projection CLS 551", **projection_diag})
    configs["study_subspace_projection"] = projection_diag

    print("training full551 split conditional CORAL")
    split_config = SplitAdapterConfig(
        inv_dim=128,
        nuisance_dim=64,
        hidden_dim=256,
        study_embedding_dim=16,
        dropout=0.05,
        epochs=120,
        warmup_epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=1e-4,
        lambda_grl=0.5,
        lambda_schedule="linear",
        reconstruction_weight=1.0,
        adversary_weight=0.1,
        distance_weight=0.05,
        variance_weight=0.05,
        conditional_coral_weight=0.05,
        seed=42,
    )
    split = train_split_cls_adapter(raw_cls, metadata, config=split_config)
    split_path = OUT_FULL / "biogpt_split_conditional_coral_zinv_551.csv"
    split_nuisance_path = OUT_FULL / "biogpt_split_conditional_coral_nuisance_551.csv"
    split.corrected_embeddings.to_csv(split_path, index=False)
    split.nuisance_embeddings.to_csv(split_nuisance_path, index=False)
    corrected_specs.append(("BiomeGPT split conditional CORAL CLS 551", split_path, False, "biogpt_split_conditional_coral_cls_551"))
    diag_rows.append({"method": "BiomeGPT split conditional CORAL CLS 551", **split.diagnostics, "n_samples": len(metadata), "n_dims": 128})
    hist = split.training_history.copy()
    hist.insert(0, "method", "BiomeGPT split conditional CORAL CLS 551")
    histories.append(hist)
    configs["split_conditional_coral"] = split.config

    rows: list[dict[str, Any]] = []
    for method, path, abundance_like, slug in corrected_specs:
        matrix = align_matrix(read_matrix(path), metadata)
        rows.extend(evaluate_method(method, matrix, metadata, abundance_like=abundance_like))
        save_pca_plot(matrix, metadata, "studyID", OUT_FIGURES / f"{slug}_python_pca_by_study.png")
        save_pca_plot(matrix, metadata, "study_condition", OUT_FIGURES / f"{slug}_python_pca_by_condition.png")
        print("evaluated", method)

    metrics = pd.DataFrame(rows)
    long_path = OUT_METRICS / "full551_biogpt_cls_metrics_long.csv"
    primary_path = OUT_METRICS / "full551_biogpt_cls_comparison.csv"
    diagnostics_path = OUT_METRICS / "full551_biogpt_cls_diagnostics.csv"
    history_path = OUT_METRICS / "full551_biogpt_cls_training_history.csv"
    config_path = OUT_METRICS / "full551_biogpt_cls_configs.json"
    metrics.to_csv(long_path, index=False)
    primary = pivot_primary(metrics)
    order = {
        "Raw abundance": 0,
        "MMUPHin adjusted abundance": 1,
        "BiomeGPT raw CLS 551": 2,
        "BiomeGPT study-mean-centered CLS 551": 3,
        "BiomeGPT NoMean conditional GRL CLS 551": 4,
        "BiomeGPT study-subspace projection CLS 551": 5,
        "BiomeGPT split conditional CORAL CLS 551": 6,
    }
    primary["_order"] = primary["method"].map(order)
    primary = primary.sort_values("_order").drop(columns="_order")
    primary.to_csv(primary_path, index=False)
    pd.DataFrame(diag_rows).to_csv(diagnostics_path, index=False)
    pd.concat(histories, ignore_index=True).to_csv(history_path, index=False)
    config_path.write_text(json.dumps(configs, indent=2), encoding="utf-8")

    raw_cls_row = primary[primary["method"] == "BiomeGPT raw CLS 551"].iloc[0]
    lines = [
        "# Full551 BiomeGPT CLS Summary",
        "",
        "This evaluates BiomeGPT CLS extracted directly for all 551 MMUPHin CRC samples, then runs local model-based CLS correction prototypes.",
        "",
        "Important: abundance rows in this CLS table use the Python representation evaluator for same-table comparison. The frozen R/Bray-Curtis canonical abundance metrics remain in `full551_raw_mmuphin_metrics.csv` and `full551_grl_abundance_comparison.csv`.",
        "",
        "## Primary Table",
        "",
        primary.to_markdown(index=False),
        "",
        "## Reading",
        "",
        f"- Raw BiomeGPT CLS study R2 is {raw_cls_row['study_R2_condition_controlled']:.4f}; study BA is {raw_cls_row['study_prediction_balanced_accuracy']:.3f}; disease LOSO AUC is {raw_cls_row['disease_LOSO_mean_within_study_AUC']:.3f}.",
        "- These CLS corrections are full-data prototypes. Do not treat them as final without LOSO/cross-fitted correction.",
        "",
        "## Outputs",
        "",
        f"- `{primary_path.relative_to(ROOT)}`",
        f"- `{diagnostics_path.relative_to(ROOT)}`",
        f"- `{OUT_FIGURES.relative_to(ROOT)}`",
    ]
    (REPORTS / "full551_biogpt_cls_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("FULL551_BIOGPT_CLS_CORRECTIONS_OK")
    print(primary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
