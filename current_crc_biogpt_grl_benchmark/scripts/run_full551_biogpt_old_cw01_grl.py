from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_mmuphin_style_crc389 import evaluate_method, pivot_primary
from src.evaluation.io import read_matrix
from src.evaluation.plots import save_pca_plot
from src.grl_correction.nomean_adapter import effective_rank, pc1_condition_auc
from src.grl_correction.train_grl import train_embedding_grl


FULL_DATA = PROJECT / "crc_controlled_benchmark" / "data"
OUT_FULL = ROOT / "outputs" / "crc_full551_benchmark"
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_FIGURES = ROOT / "outputs" / "figures" / "full551_biogpt_old_cw01_grl"
REPORTS = ROOT / "reports"


def align_matrix(matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    ids = metadata["sample_id"].astype(str).tolist()
    out = matrix.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    return out.set_index("sample_id").loc[ids].reset_index()


def diagnostics(matrix: pd.DataFrame, metadata: pd.DataFrame, method: str) -> dict[str, Any]:
    x = matrix.drop(columns=["sample_id"]).astype(float).to_numpy()
    return {
        "method": method,
        "n_samples": int(x.shape[0]),
        "n_dims": int(x.shape[1]),
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
    raw_cls = align_matrix(read_matrix(OUT_FULL / "biogpt_raw_cls_551.csv"), metadata)
    print("training BiomeGPT CLS old cw0.1-style GRL")
    result = train_embedding_grl(
        raw_cls,
        metadata,
        latent_dim=8,
        hidden_dim=128,
        epochs=100,
        lr=1e-3,
        lambda_grl=10.0,
        lambda_schedule="linear",
        warmup_fraction=0.1,
        condition_weight=0.1,
        preserve_weight=0.001,
        use_class_weights=True,
        batch_size=64,
        shuffle=True,
        seed=42,
        condition_aware_adversary=True,
        condition_embedding_dim=16,
        use_study_conditioned_decoder=False,
        external_eval_every=20,
        device="cpu",
    )
    corrected_path = OUT_FULL / "biogpt_old_cw01_grl_z8_cls_551.csv"
    history_path = OUT_METRICS / "full551_biogpt_old_cw01_grl_training_history.csv"
    probes_path = OUT_METRICS / "full551_biogpt_old_cw01_grl_internal_probes.csv"
    config_path = OUT_METRICS / "full551_biogpt_old_cw01_grl_config.json"
    result.corrected_embeddings.to_csv(corrected_path, index=False)
    result.history.to_csv(history_path, index=False)
    result.final_probe_metrics.to_csv(probes_path, index=False)
    config_path.write_text(json.dumps(result.config, indent=2), encoding="utf-8")

    methods = [
        ("Raw abundance", FULL_DATA / "crc_raw_abundance.csv", True, "raw_abundance"),
        ("MMUPHin adjusted abundance", FULL_DATA / "crc_mmuphin_adjusted_abundance.csv", True, "mmuphin_adjusted_abundance"),
        ("BiomeGPT raw CLS 551", OUT_FULL / "biogpt_raw_cls_551.csv", False, "biogpt_raw_cls_551"),
        ("BiomeGPT study-mean-centered CLS 551", OUT_FULL / "biogpt_study_mean_centered_cls_551.csv", False, "biogpt_mean_centered_cls_551"),
        ("BiomeGPT old cw0.1-style GRL CLS 551", corrected_path, False, "biogpt_old_cw01_grl_cls_551"),
    ]
    optional = [
        ("BiomeGPT NoMean conditional GRL CLS 551", OUT_FULL / "biogpt_nomean_conditional_grl_cls_551.csv", False, "biogpt_nomean_conditional_grl_cls_551"),
        ("BiomeGPT split conditional CORAL CLS 551", OUT_FULL / "biogpt_split_conditional_coral_zinv_551.csv", False, "biogpt_split_conditional_coral_cls_551"),
    ]
    methods.extend(item for item in optional if item[1].exists())

    rows: list[dict[str, Any]] = []
    diag_rows = []
    for method, path, abundance_like, slug in methods:
        matrix = align_matrix(read_matrix(path), metadata)
        rows.extend(evaluate_method(method, matrix, metadata, abundance_like=abundance_like))
        if not abundance_like:
            diag_rows.append(diagnostics(matrix, metadata, method))
        save_pca_plot(matrix, metadata, "studyID", OUT_FIGURES / f"{slug}_python_pca_by_study.png")
        save_pca_plot(matrix, metadata, "study_condition", OUT_FIGURES / f"{slug}_python_pca_by_condition.png")
        print("evaluated", method)

    metrics = pd.DataFrame(rows)
    primary = pivot_primary(metrics)
    order = {
        "Raw abundance": 0,
        "MMUPHin adjusted abundance": 1,
        "BiomeGPT raw CLS 551": 2,
        "BiomeGPT study-mean-centered CLS 551": 3,
        "BiomeGPT old cw0.1-style GRL CLS 551": 4,
        "BiomeGPT NoMean conditional GRL CLS 551": 5,
        "BiomeGPT split conditional CORAL CLS 551": 6,
    }
    primary["_order"] = primary["method"].map(order)
    primary = primary.sort_values("_order").drop(columns="_order")
    primary_path = OUT_METRICS / "full551_biogpt_old_cw01_grl_comparison.csv"
    metrics_long_path = OUT_METRICS / "full551_biogpt_old_cw01_grl_metrics_long.csv"
    diagnostics_path = OUT_METRICS / "full551_biogpt_old_cw01_grl_diagnostics.csv"
    primary.to_csv(primary_path, index=False)
    metrics.to_csv(metrics_long_path, index=False)
    pd.DataFrame(diag_rows).to_csv(diagnostics_path, index=False)

    old_row = primary[primary["method"] == "BiomeGPT old cw0.1-style GRL CLS 551"].iloc[0]
    raw_row = primary[primary["method"] == "BiomeGPT raw CLS 551"].iloc[0]
    lines = [
        "# Full551 BiomeGPT Old cw0.1-Style GRL",
        "",
        "This runs the old supervised cw0.1 GRL idea on top of BiomeGPT raw CLS embeddings for the canonical 551 MMUPHin CRC samples. It is an embedding-level correction prototype, not foundation-model pretraining.",
        "",
        "Config summary: `latent_dim=8`, `lambda_grl=10`, `condition_weight=0.1`, `preserve_weight=0.001`, linear warmup, condition-aware study adversary.",
        "",
        "## Primary Table",
        "",
        primary.to_markdown(index=False),
        "",
        "## Reading",
        "",
        f"- Raw CLS study BA {raw_row['study_prediction_balanced_accuracy']:.3f}, disease LOSO AUC {raw_row['disease_LOSO_mean_within_study_AUC']:.3f}, study R2 {raw_row['study_R2_condition_controlled']:.4f}.",
        f"- Old cw0.1-style CLS GRL study BA {old_row['study_prediction_balanced_accuracy']:.3f}, disease LOSO AUC {old_row['disease_LOSO_mean_within_study_AUC']:.3f}, study R2 {old_row['study_R2_condition_controlled']:.4f}.",
        "- Because this uses a condition classifier and an 8-dimensional bottleneck, inspect effective rank and PC1 condition AUC before interpreting any disease AUC improvement.",
        "- This is full-data/transductive correction. It cannot support a final claim without LOSO/cross-fitted correction.",
        "",
        "## Outputs",
        "",
        f"- `{corrected_path.relative_to(ROOT)}`",
        f"- `{primary_path.relative_to(ROOT)}`",
        f"- `{diagnostics_path.relative_to(ROOT)}`",
        f"- `{OUT_FIGURES.relative_to(ROOT)}`",
    ]
    (REPORTS / "full551_biogpt_old_cw01_grl_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("FULL551_BIOGPT_OLD_CW01_GRL_OK")
    print(primary.to_string(index=False))
    print(pd.DataFrame(diag_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
