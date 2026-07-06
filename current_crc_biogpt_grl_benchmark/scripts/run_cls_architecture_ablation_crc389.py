from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_biogpt_cls_crc389 import align_matrix, study_mean_center_cls
from scripts.evaluate_mmuphin_style_crc389 import evaluate_method, pivot_primary
from src.evaluation.io import read_matrix
from src.evaluation.plots import save_pca_plot
from src.grl_correction.nomean_adapter import effective_rank, pc1_condition_auc, standardize_matrix
from src.grl_correction.split_adapter import SplitAdapterConfig, train_split_cls_adapter


DATA_DIR = ROOT / "outputs" / "crc_overlap_benchmark"
METRIC_DIR = ROOT / "outputs" / "metrics"
FIGURE_DIR = ROOT / "outputs" / "figures" / "cls_architecture_ablation_crc389"
REPORT_DIR = ROOT / "reports"


def study_subspace_projection(raw_cls: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for study-subspace projection.") from exc

    raw_cls = align_matrix(raw_cls, metadata)
    x, _, _, feature_cols, sample_ids = standardize_matrix(raw_cls)
    y = metadata["studyID"].astype(str).to_numpy()
    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x, y)
    weights = model.coef_.astype(float)
    weights = weights - weights.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(weights, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-8))
    basis = vt[:rank].T if rank > 0 else np.zeros((x.shape[1], 0))
    projected = x - (x @ basis) @ basis.T if basis.shape[1] else x.copy()
    out = pd.DataFrame(projected, columns=feature_cols)
    out.insert(0, "sample_id", sample_ids)
    diagnostics = {
        "effective_rank": effective_rank(projected),
        "raw_effective_rank": effective_rank(x),
        "pc1_condition_auc": pc1_condition_auc(projected, metadata),
        "raw_pc1_condition_auc": pc1_condition_auc(x, metadata),
        "mean_squared_shift_standardized": float(np.mean((projected - x) ** 2)),
        "mean_l2_shift_standardized": float(np.mean(np.linalg.norm(projected - x, axis=1))),
        "output_dim": int(projected.shape[1]),
        "removed_subspace_rank": rank,
        "study_classifier_training_accuracy": float(model.score(x, y)),
    }
    return out, diagnostics


def split_configs() -> list[tuple[str, str, SplitAdapterConfig]]:
    base = {
        "inv_dim": 128,
        "nuisance_dim": 64,
        "hidden_dim": 256,
        "study_embedding_dim": 16,
        "dropout": 0.05,
        "epochs": 120,
        "warmup_epochs": 20,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "lambda_grl": 0.50,
        "lambda_schedule": "linear",
        "reconstruction_weight": 1.0,
        "adversary_weight": 0.1,
        "distance_weight": 0.05,
        "variance_weight": 0.05,
        "use_class_weights": True,
        "seed": 42,
    }
    return [
        (
            "G invariant/nuisance split + weak GRL",
            "cls_arch_G_split_weak_grl",
            SplitAdapterConfig(**base, conditional_coral_weight=0.0),
        ),
        (
            "I invariant/nuisance split + conditional CORAL",
            "cls_arch_I_split_grl_conditional_coral",
            SplitAdapterConfig(**base, conditional_coral_weight=0.05),
        ),
    ]


def method_specs(extra_paths: list[tuple[str, Path, str]]) -> list[tuple[str, Path, bool, str]]:
    specs = [
        ("Raw abundance", DATA_DIR / "raw_abundance_389.csv", True, "raw_abundance"),
        ("MMUPHin adjusted abundance", DATA_DIR / "mmuphin_adjusted_abundance_389.csv", True, "mmuphin_adjusted_abundance"),
        ("BiomeGPT raw CLS", DATA_DIR / "biogpt_raw_cls_389.csv", False, "biogpt_raw_cls"),
        ("BiomeGPT study-mean-centered CLS", DATA_DIR / "biogpt_mean_centered_cls_389.csv", False, "biogpt_mean_centered_cls"),
    ]
    previous = [
        ("A NoMean adapter + vanilla GRL", DATA_DIR / "nomean_ablation_A_vanilla_grl.csv", False, "A_nomean_vanilla_grl"),
        ("B NoMean adapter + conditional GRL", DATA_DIR / "nomean_ablation_B_conditional_grl.csv", False, "B_nomean_conditional_grl"),
    ]
    specs.extend(item for item in previous if item[1].exists())
    specs.extend((name, path, False, slug) for name, path, slug in extra_paths)
    return specs


def write_summary(primary: pd.DataFrame, diagnostics: pd.DataFrame, paths: dict[str, str]) -> None:
    raw_cls = primary[primary["method"] == "BiomeGPT raw CLS"].iloc[0]
    mmuphin = primary[primary["method"] == "MMUPHin adjusted abundance"].iloc[0]
    candidates = primary[primary["method"].str.match(r"^[GHI] ")].copy()
    candidates["batch_reduction_vs_raw_cls"] = (
        raw_cls["study_prediction_balanced_accuracy"] - candidates["study_prediction_balanced_accuracy"]
    )
    candidates["study_r2_reduction_vs_raw_cls"] = (
        raw_cls["study_R2_condition_controlled"] - candidates["study_R2_condition_controlled"]
    )
    candidates["biology_change_vs_raw_cls"] = (
        candidates["disease_LOSO_mean_within_study_AUC"] - raw_cls["disease_LOSO_mean_within_study_AUC"]
    )
    candidates["biology_retention_vs_raw_cls"] = (
        candidates["disease_LOSO_mean_within_study_AUC"] / raw_cls["disease_LOSO_mean_within_study_AUC"]
    )
    useful = candidates[
        (candidates["study_R2_condition_controlled"] < raw_cls["study_R2_condition_controlled"])
        & (candidates["study_prediction_balanced_accuracy"] < raw_cls["study_prediction_balanced_accuracy"])
        & (candidates["disease_LOSO_mean_within_study_AUC"] >= raw_cls["disease_LOSO_mean_within_study_AUC"] - 0.05)
    ].copy()
    if useful.empty:
        verdict = "None of H/G/I met the full useful rule of lowering both study R2 and study BA while preserving disease AUC within 0.05 of raw CLS."
    else:
        picked = useful.sort_values(["study_R2_condition_controlled", "study_prediction_balanced_accuracy"]).iloc[0]
        verdict = (
            f"{picked['method']} is the best current H/G/I candidate under the useful rule "
            f"(study R2 {picked['study_R2_condition_controlled']:.4f}, study BA {picked['study_prediction_balanced_accuracy']:.3f}, disease AUC {picked['disease_LOSO_mean_within_study_AUC']:.3f})."
        )

    lines = [
        "# CLS Architecture Ablation CRC389",
        "",
        "## Scope",
        "",
        "This run tests architecture changes after the NoMean residual adapter failed to move study R2 enough. It does not train BiomeGPT, does not apply study mean-centering inside the tested methods, and does not optimize final evaluator metrics directly.",
        "",
        "Tested candidates:",
        "",
        "- H: fixed linear study-subspace projection from raw CLS",
        "- G: invariant/nuisance split adapter with weak conditional GRL",
        "- I: invariant/nuisance split adapter with weak conditional GRL plus conditional CORAL",
        "",
        "## Primary Table",
        "",
        primary.to_markdown(index=False),
        "",
        "## H/G/I Tradeoff Table",
        "",
        candidates.to_markdown(index=False),
        "",
        "## Diagnostics",
        "",
        diagnostics.to_markdown(index=False),
        "",
        "## Reading",
        "",
        f"- Raw CLS: study R2 {raw_cls['study_R2_condition_controlled']:.4f}, study BA {raw_cls['study_prediction_balanced_accuracy']:.3f}, disease LOSO AUC {raw_cls['disease_LOSO_mean_within_study_AUC']:.3f}.",
        f"- MMUPHin adjusted abundance reference: study R2 {mmuphin['study_R2_condition_controlled']:.4f}, study BA {mmuphin['study_prediction_balanced_accuracy']:.3f}, disease LOSO AUC {mmuphin['disease_LOSO_mean_within_study_AUC']:.3f}.",
        f"- Main verdict: {verdict}",
        "- H is the most direct diagnostic of whether raw CLS study structure is carried by linear study-discriminative directions.",
        "- G/I are full-data prototypes. If one looks promising, the next required step is LOSO/cross-fitted correction.",
        "",
        "## Output Files",
        "",
    ]
    for key, value in paths.items():
        lines.append(f"- `{key}`: `{Path(value).relative_to(ROOT)}`")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cls_architecture_ablation_crc389_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    extra_paths: list[tuple[str, Path, str]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    config_dump: dict[str, Any] = {}

    print("building H study-subspace projection")
    h_matrix, h_diag = study_subspace_projection(raw_cls, metadata)
    h_path = DATA_DIR / "cls_arch_H_study_subspace_projection.csv"
    h_matrix.to_csv(h_path, index=False)
    extra_paths.append(("H study-subspace projection from raw CLS", h_path, "H_study_subspace_projection"))
    diagnostics_rows.append({"method": "H study-subspace projection from raw CLS", **h_diag})
    config_dump["cls_arch_H_study_subspace_projection"] = {"method": "linear study classifier weight subspace projection", **h_diag}

    for name, slug, config in split_configs():
        print("training", name)
        result = train_split_cls_adapter(raw_cls, metadata, config=config)
        corrected_path = DATA_DIR / f"{slug}.csv"
        nuisance_path = DATA_DIR / f"{slug}_nuisance.csv"
        result.corrected_embeddings.to_csv(corrected_path, index=False)
        result.nuisance_embeddings.to_csv(nuisance_path, index=False)
        extra_paths.append((name, corrected_path, slug))
        diagnostics_rows.append({"method": name, **result.diagnostics})
        hist = result.training_history.copy()
        hist.insert(0, "method", name)
        hist.insert(1, "slug", slug)
        histories.append(hist)
        config_dump[slug] = result.config | {"label_maps": result.label_maps, "nuisance_path": str(nuisance_path)}

    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    diagnostics = pd.DataFrame(diagnostics_rows)
    history_path = METRIC_DIR / "cls_architecture_ablation_crc389_training_history.csv"
    diagnostics_path = METRIC_DIR / "cls_architecture_ablation_crc389_diagnostics.csv"
    config_path = METRIC_DIR / "cls_architecture_ablation_crc389_configs.json"
    history.to_csv(history_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    config_path.write_text(json.dumps(config_dump, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for method, path, abundance_like, slug in method_specs(extra_paths):
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
        "H study-subspace projection from raw CLS": 6,
        "G invariant/nuisance split + weak GRL": 7,
        "I invariant/nuisance split + conditional CORAL": 8,
    }
    primary["_order"] = primary["method"].map(order)
    primary = primary.sort_values("_order").drop(columns="_order")
    long_path = METRIC_DIR / "cls_architecture_ablation_crc389_metrics_long.csv"
    primary_path = METRIC_DIR / "cls_architecture_ablation_crc389_primary_table.csv"
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
    print("CLS_ARCHITECTURE_ABLATION_CRC389_OK")
    print(primary.to_string(index=False))
    print(diagnostics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
