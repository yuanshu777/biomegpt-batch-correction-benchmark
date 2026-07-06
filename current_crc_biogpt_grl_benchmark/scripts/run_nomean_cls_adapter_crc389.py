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
FIGURE_DIR = ROOT / "outputs" / "figures" / "nomean_cls_adapter_crc389"
REPORT_DIR = ROOT / "reports"


def adapter_configs() -> list[tuple[str, str, NoMeanAdapterConfig]]:
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
            "NoMean CLS adapter preserve-only",
            "nomean_cls_adapter_preserve_only",
            NoMeanAdapterConfig(**base, lambda_grl=0.0, adversary_weight=0.0),
        ),
        (
            "NoMean CLS adapter weak conditional GRL 0.05",
            "nomean_cls_adapter_lam005",
            NoMeanAdapterConfig(**base, lambda_grl=0.05, adversary_weight=0.1),
        ),
        (
            "NoMean CLS adapter weak conditional GRL 0.10",
            "nomean_cls_adapter_lam010",
            NoMeanAdapterConfig(**base, lambda_grl=0.10, adversary_weight=0.1),
        ),
        (
            "NoMean CLS adapter weak conditional GRL 0.30",
            "nomean_cls_adapter_lam030",
            NoMeanAdapterConfig(**base, lambda_grl=0.30, adversary_weight=0.1),
        ),
        (
            "NoMean CLS adapter weak conditional GRL 0.50",
            "nomean_cls_adapter_lam050",
            NoMeanAdapterConfig(**base, lambda_grl=0.50, adversary_weight=0.1),
        ),
        (
            "NoMean CLS adapter weak conditional GRL 1.00",
            "nomean_cls_adapter_lam100",
            NoMeanAdapterConfig(**base, lambda_grl=1.00, adversary_weight=0.1),
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
    centered = primary[primary["method"] == "BiomeGPT study-mean-centered CLS"].iloc[0]
    adapter_rows = primary[primary["method"].str.startswith("NoMean CLS adapter")].copy()
    best_study = adapter_rows.sort_values("study_prediction_balanced_accuracy").iloc[0]
    best_auc = adapter_rows.sort_values("disease_LOSO_mean_within_study_AUC", ascending=False).iloc[0]
    useful_rows = adapter_rows[
        (adapter_rows["study_prediction_balanced_accuracy"] < raw_cls["study_prediction_balanced_accuracy"])
        & (adapter_rows["disease_LOSO_mean_within_study_AUC"] >= raw_cls["disease_LOSO_mean_within_study_AUC"] - 0.05)
    ].copy()
    if useful_rows.empty:
        verdict = (
            "No adapter configuration met the minimal useful rule of reducing study BA versus raw CLS while keeping disease LOSO AUC within 0.05 of raw CLS."
        )
    else:
        picked = useful_rows.sort_values("study_prediction_balanced_accuracy").iloc[0]
        verdict = (
            f"{picked['method']} met the minimal useful rule: lower study BA than raw CLS and disease LOSO AUC within 0.05 of raw CLS."
        )

    lines = [
        "# NoMean BiomeGPT CLS Adapter CRC389 Prototype",
        "",
        "## Scope",
        "",
        "This is a local prototype on the 389 overlap samples. It does not train BiomeGPT, does not run scGPT pretraining, and does not apply manual study mean-centering inside the adapter method.",
        "",
        "The adapter is a small residual module on raw BiomeGPT CLS: `z = h + 0.1 * Adapter(h)`. It keeps the CLS dimension at 512 and uses a study-aware decoder side-channel plus weak conditional study GRL. No condition classifier is used.",
        "",
        "## Primary Table",
        "",
        primary.to_markdown(index=False),
        "",
        "## Adapter Diagnostics",
        "",
        diagnostics.to_markdown(index=False),
        "",
        "## Reading",
        "",
        f"- Raw CLS study BA is {raw_cls['study_prediction_balanced_accuracy']:.3f}; mean-centering study BA is {centered['study_prediction_balanced_accuracy']:.3f}.",
        f"- Best adapter by study BA is `{best_study['method']}` with study BA {best_study['study_prediction_balanced_accuracy']:.3f} and disease LOSO AUC {best_study['disease_LOSO_mean_within_study_AUC']:.3f}.",
        f"- Best adapter by disease LOSO AUC is `{best_auc['method']}` with disease LOSO AUC {best_auc['disease_LOSO_mean_within_study_AUC']:.3f} and study BA {best_auc['study_prediction_balanced_accuracy']:.3f}.",
        f"- Main verdict: {verdict}",
        "- This is still full-data/transductive adapter fitting. If a configuration looks promising, the next strict check is LOSO/cross-fitted adapter correction.",
        "- Do not claim it beats MMUPHin unless it survives the same-sample evaluator and strict held-out correction.",
        "",
        "## Output Files",
        "",
    ]
    for key, value in paths.items():
        lines.append(f"- `{key}`: `{Path(value).relative_to(ROOT)}`")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "nomean_cls_adapter_crc389_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    diagnostics_rows: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    config_dump: dict[str, Any] = {}
    for name, slug, config in adapter_configs():
        print("training", name)
        result = train_nomean_cls_adapter(raw_cls, metadata, config=config)
        corrected_path = DATA_DIR / f"{slug}.csv"
        result.corrected_embeddings.to_csv(corrected_path, index=False)
        adapter_paths.append((name, corrected_path, slug))
        diagnostics_rows.append({"method": name, **result.diagnostics})
        hist = result.training_history.copy()
        hist.insert(0, "method", name)
        hist.insert(1, "slug", slug)
        histories.append(hist)
        config_dump[slug] = result.config | {"label_maps": result.label_maps}

    history = pd.concat(histories, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostics_rows)
    history_path = METRIC_DIR / "nomean_cls_adapter_crc389_training_history.csv"
    diagnostics_path = METRIC_DIR / "nomean_cls_adapter_crc389_diagnostics.csv"
    config_path = METRIC_DIR / "nomean_cls_adapter_crc389_configs.json"
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
        "NoMean CLS adapter preserve-only": 4,
        "NoMean CLS adapter weak conditional GRL 0.05": 5,
        "NoMean CLS adapter weak conditional GRL 0.10": 6,
        "NoMean CLS adapter weak conditional GRL 0.30": 7,
        "NoMean CLS adapter weak conditional GRL 0.50": 8,
        "NoMean CLS adapter weak conditional GRL 1.00": 9,
    }
    primary["_order"] = primary["method"].map(order)
    primary = primary.sort_values("_order").drop(columns="_order")
    long_path = METRIC_DIR / "nomean_cls_adapter_crc389_metrics_long.csv"
    primary_path = METRIC_DIR / "nomean_cls_adapter_crc389_primary_table.csv"
    metrics.to_csv(long_path, index=False)
    primary.to_csv(primary_path, index=False)

    paths = {
        "corrected_adapter_dir": str(DATA_DIR),
        "metrics_long": str(long_path),
        "primary_table": str(primary_path),
        "training_history": str(history_path),
        "diagnostics": str(diagnostics_path),
        "configs": str(config_path),
        "figure_dir": str(FIGURE_DIR),
    }
    write_summary(primary, diagnostics, paths)
    print("NOMEAN_CLS_ADAPTER_CRC389_OK")
    print(primary.to_string(index=False))
    print(diagnostics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
