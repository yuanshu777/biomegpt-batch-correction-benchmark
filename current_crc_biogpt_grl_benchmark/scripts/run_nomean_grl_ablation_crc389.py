from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_biogpt_cls_crc389 import align_matrix, study_mean_center_cls
from scripts.evaluate_mmuphin_style_crc389 import evaluate_method, pivot_primary
from src.evaluation.io import read_matrix
from src.evaluation.plots import save_pca_plot
from src.grl_correction.nomean_adapter import NoMeanAdapterConfig, train_nomean_cls_adapter


DATA_DIR = ROOT / "outputs" / "crc_overlap_benchmark"
METRIC_DIR = ROOT / "outputs" / "metrics"
FIGURE_DIR = ROOT / "outputs" / "figures" / "nomean_grl_ablation_crc389"
REPORT_DIR = ROOT / "reports"


def ablation_configs() -> list[tuple[str, str, NoMeanAdapterConfig]]:
    base = {
        "hidden_dim": 128,
        "study_embedding_dim": 16,
        "residual_scale": 0.1,
        "dropout": 0.05,
        "epochs": 100,
        "warmup_epochs": 20,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "lambda_grl": 0.50,
        "lambda_schedule": "linear",
        "reconstruction_weight": 1.0,
        "preserve_weight": 1.0,
        "variance_weight": 0.02,
        "covariance_weight": 0.005,
        "use_class_weights": True,
        "seed": 42,
    }
    return [
        (
            "A NoMean adapter + vanilla GRL",
            "nomean_ablation_A_vanilla_grl",
            NoMeanAdapterConfig(
                **base,
                adversary_mode="vanilla",
                adversary_weight=0.1,
                condition_prior_weight=0.0,
            ),
        ),
        (
            "B NoMean adapter + conditional GRL",
            "nomean_ablation_B_conditional_grl",
            NoMeanAdapterConfig(
                **base,
                adversary_mode="conditional",
                adversary_weight=0.1,
                condition_prior_weight=0.0,
            ),
        ),
        (
            "C NoMean adapter + residual conditional GRL",
            "nomean_ablation_C_residual_conditional_grl",
            NoMeanAdapterConfig(
                **base,
                adversary_mode="residual_conditional",
                adversary_weight=0.1,
                condition_prior_weight=0.1,
            ),
        ),
        (
            "D NoMean adapter + pairwise within-condition GRL",
            "nomean_ablation_D_pairwise_within_condition_grl",
            NoMeanAdapterConfig(
                **base,
                adversary_mode="pairwise_within_condition",
                adversary_weight=0.0,
                condition_prior_weight=0.0,
                pairwise_weight=0.1,
            ),
        ),
        (
            "E NoMean adapter + residual conditional GRL + conditional CORAL",
            "nomean_ablation_E_residual_conditional_grl_coral",
            NoMeanAdapterConfig(
                **base,
                adversary_mode="residual_conditional",
                adversary_weight=0.1,
                condition_prior_weight=0.1,
                conditional_coral_weight=0.05,
            ),
        ),
    ]


def method_specs(adapter_paths: list[tuple[str, Path, str]]) -> list[tuple[str, Path, bool, str]]:
    specs = [
        ("Raw abundance", DATA_DIR / "raw_abundance_389.csv", True, "raw_abundance"),
        ("MMUPHin adjusted abundance", DATA_DIR / "mmuphin_adjusted_abundance_389.csv", True, "mmuphin_adjusted_abundance"),
        ("BiomeGPT raw CLS", DATA_DIR / "biogpt_raw_cls_389.csv", False, "biogpt_raw_cls"),
        ("BiomeGPT study-mean-centered CLS", DATA_DIR / "biogpt_mean_centered_cls_389.csv", False, "biogpt_mean_centered_cls"),
    ]
    specs.extend((name, path, False, slug) for name, path, slug in adapter_paths)
    return specs


def write_summary(primary: pd.DataFrame, diagnostics: pd.DataFrame, paths: dict[str, str]) -> None:
    raw_cls = primary[primary["method"] == "BiomeGPT raw CLS"].iloc[0]
    mmuphin = primary[primary["method"] == "MMUPHin adjusted abundance"].iloc[0]
    ablations = primary[primary["method"].str.match(r"^[A-E] ")].copy()
    ablations["batch_reduction_vs_raw_cls"] = (
        raw_cls["study_prediction_balanced_accuracy"] - ablations["study_prediction_balanced_accuracy"]
    )
    ablations["biology_change_vs_raw_cls"] = (
        ablations["disease_LOSO_mean_within_study_AUC"] - raw_cls["disease_LOSO_mean_within_study_AUC"]
    )
    ablations["biology_retention_vs_raw_cls"] = (
        ablations["disease_LOSO_mean_within_study_AUC"] / raw_cls["disease_LOSO_mean_within_study_AUC"]
    )
    best_study = ablations.sort_values("study_prediction_balanced_accuracy").iloc[0]
    best_tradeoff = ablations[
        ablations["disease_LOSO_mean_within_study_AUC"] >= raw_cls["disease_LOSO_mean_within_study_AUC"] - 0.05
    ].sort_values("study_prediction_balanced_accuracy")
    if best_tradeoff.empty:
        verdict = "No ablation preserved raw CLS disease LOSO AUC within 0.05 while reducing study BA."
    else:
        picked = best_tradeoff.iloc[0]
        verdict = (
            f"{picked['method']} is the best current tradeoff under the raw-CLS preservation rule "
            f"(study BA {picked['study_prediction_balanced_accuracy']:.3f}, disease LOSO AUC {picked['disease_LOSO_mean_within_study_AUC']:.3f})."
        )

    lines = [
        "# NoMean CLS GRL Formulation Ablation CRC389",
        "",
        "## Scope",
        "",
        "This run compares five GRL formulations on the same NoMean residual BiomeGPT CLS adapter. It does not train BiomeGPT, does not use mean-centering inside the adapter, and does not optimize the final evaluator metrics directly.",
        "",
        "All ablations use the same local prototype settings: 512-dimensional CLS output, residual scale 0.1, study-aware decoder reconstruction, preservation loss, anti-collapse variance/covariance penalties, and `lambda_grl=0.50` with linear warmup.",
        "",
        "## Primary Table",
        "",
        primary.to_markdown(index=False),
        "",
        "## Tradeoff Table",
        "",
        ablations.to_markdown(index=False),
        "",
        "## Adapter Diagnostics",
        "",
        diagnostics.to_markdown(index=False),
        "",
        "## Reading",
        "",
        f"- Raw CLS: study BA {raw_cls['study_prediction_balanced_accuracy']:.3f}, study R2 {raw_cls['study_R2_condition_controlled']:.4f}, disease LOSO AUC {raw_cls['disease_LOSO_mean_within_study_AUC']:.3f}.",
        f"- MMUPHin adjusted abundance reference: study BA {mmuphin['study_prediction_balanced_accuracy']:.3f}, study R2 {mmuphin['study_R2_condition_controlled']:.4f}, disease LOSO AUC {mmuphin['disease_LOSO_mean_within_study_AUC']:.3f}.",
        f"- Lowest study BA among ablations: {best_study['method']} with study BA {best_study['study_prediction_balanced_accuracy']:.3f}.",
        f"- Main verdict: {verdict}",
        "- This is still full-data/transductive adapter fitting. Any promising formulation needs LOSO/cross-fitted adapter correction before a scientific claim.",
        "",
        "## Output Files",
        "",
    ]
    for key, value in paths.items():
        lines.append(f"- `{key}`: `{Path(value).relative_to(ROOT)}`")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "nomean_grl_ablation_crc389_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(DATA_DIR / "metadata_389.csv", dtype=str)
    raw_cls = align_matrix(read_matrix(DATA_DIR / "biogpt_raw_cls_389.csv"), metadata)
    centered_path = DATA_DIR / "biogpt_mean_centered_cls_389.csv"
    if not centered_path.exists():
        study_mean_center_cls(raw_cls, metadata).to_csv(centered_path, index=False)

    adapter_paths: list[tuple[str, Path, str]] = []
    histories: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, Any]] = []
    config_dump: dict[str, Any] = {}
    for name, slug, config in ablation_configs():
        print("training", name)
        result = train_nomean_cls_adapter(raw_cls, metadata, config=config)
        corrected_path = DATA_DIR / f"{slug}.csv"
        result.corrected_embeddings.to_csv(corrected_path, index=False)
        adapter_paths.append((name, corrected_path, slug))
        hist = result.training_history.copy()
        hist.insert(0, "method", name)
        hist.insert(1, "slug", slug)
        histories.append(hist)
        diagnostics_rows.append({"method": name, **result.diagnostics})
        config_dump[slug] = result.config | {"label_maps": result.label_maps}

    history = pd.concat(histories, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostics_rows)
    history_path = METRIC_DIR / "nomean_grl_ablation_crc389_training_history.csv"
    diagnostics_path = METRIC_DIR / "nomean_grl_ablation_crc389_diagnostics.csv"
    config_path = METRIC_DIR / "nomean_grl_ablation_crc389_configs.json"
    history.to_csv(history_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    config_path.write_text(json.dumps(config_dump, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for method, path, abundance_like, slug in method_specs(adapter_paths):
        matrix = align_matrix(read_matrix(path), metadata)
        rows.extend(evaluate_method(method, matrix, metadata, abundance_like=abundance_like))
        save_pca_plot(matrix, metadata, "studyID", FIGURE_DIR / f"{slug}_pca_by_study.png")
        save_pca_plot(matrix, metadata, "study_condition", FIGURE_DIR / f"{slug}_pca_by_condition.png")
        print("evaluated", method)

    metrics = pd.DataFrame(rows)
    primary = pivot_primary(metrics)
    order = {
        "Raw abundance": 0,
        "MMUPHin adjusted abundance": 1,
        "BiomeGPT raw CLS": 2,
        "BiomeGPT study-mean-centered CLS": 3,
        "A NoMean adapter + vanilla GRL": 4,
        "B NoMean adapter + conditional GRL": 5,
        "C NoMean adapter + residual conditional GRL": 6,
        "D NoMean adapter + pairwise within-condition GRL": 7,
        "E NoMean adapter + residual conditional GRL + conditional CORAL": 8,
    }
    primary["_order"] = primary["method"].map(order)
    primary = primary.sort_values("_order").drop(columns="_order")
    long_path = METRIC_DIR / "nomean_grl_ablation_crc389_metrics_long.csv"
    primary_path = METRIC_DIR / "nomean_grl_ablation_crc389_primary_table.csv"
    metrics.to_csv(long_path, index=False)
    primary.to_csv(primary_path, index=False)

    paths = {
        "metrics_long": str(long_path),
        "primary_table": str(primary_path),
        "training_history": str(history_path),
        "diagnostics": str(diagnostics_path),
        "configs": str(config_path),
        "figure_dir": str(FIGURE_DIR),
        "corrected_adapter_dir": str(DATA_DIR),
    }
    write_summary(primary, diagnostics, paths)
    print("NOMEAN_GRL_ABLATION_CRC389_OK")
    print(primary.to_string(index=False))
    print(diagnostics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
