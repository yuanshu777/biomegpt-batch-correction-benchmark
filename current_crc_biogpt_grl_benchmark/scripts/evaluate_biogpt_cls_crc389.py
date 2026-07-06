from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_mmuphin_style_crc389 import evaluate_method, pivot_primary
from src.evaluation.io import read_matrix
from src.evaluation.plots import save_pca_plot


DATA_DIR = ROOT / "outputs" / "crc_overlap_benchmark"
METRIC_DIR = ROOT / "outputs" / "metrics"
FIGURE_DIR = ROOT / "outputs" / "figures" / "biogpt_cls_crc389"
REPORT_DIR = ROOT / "reports"


def align_matrix(matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    ids = metadata["sample_id"].astype(str).tolist()
    matrix = matrix.copy()
    matrix["sample_id"] = matrix["sample_id"].astype(str)
    return matrix.set_index("sample_id").loc[ids].reset_index()


def study_mean_center_cls(cls: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    aligned = align_matrix(cls, metadata)
    feature_cols = [c for c in aligned.columns if c != "sample_id"]
    joined = metadata[["sample_id", "studyID"]].merge(aligned, on="sample_id", how="inner")
    x = joined[feature_cols].astype(float)
    grand_mean = x.mean(axis=0)
    study_means = joined.groupby("studyID")[feature_cols].transform("mean")
    centered = x - study_means + grand_mean
    out = pd.DataFrame(centered.to_numpy(), columns=feature_cols)
    out.insert(0, "sample_id", joined["sample_id"].astype(str).to_numpy())
    return align_matrix(out, metadata)


def method_specs() -> list[tuple[str, Path, bool, str]]:
    return [
        ("Raw abundance", DATA_DIR / "raw_abundance_389.csv", True, "raw_abundance"),
        ("MMUPHin adjusted abundance", DATA_DIR / "mmuphin_adjusted_abundance_389.csv", True, "mmuphin_adjusted_abundance"),
        ("BiomeGPT raw CLS", DATA_DIR / "biogpt_raw_cls_389.csv", False, "biogpt_raw_cls"),
        ("BiomeGPT study-mean-centered CLS", DATA_DIR / "biogpt_mean_centered_cls_389.csv", False, "biogpt_mean_centered_cls"),
    ]


def write_summary(metrics: pd.DataFrame, paths: dict[str, str]) -> None:
    primary = pivot_primary(metrics)
    # Preserve the intended method order for CLS report.
    order = {
        "Raw abundance": 0,
        "MMUPHin adjusted abundance": 1,
        "BiomeGPT raw CLS": 2,
        "BiomeGPT study-mean-centered CLS": 3,
    }
    primary["_order"] = primary["method"].map(order)
    primary = primary.sort_values("_order").drop(columns="_order")
    raw_cls = primary[primary["method"] == "BiomeGPT raw CLS"].iloc[0]
    centered = primary[primary["method"] == "BiomeGPT study-mean-centered CLS"].iloc[0]
    lines = [
        "# BiomeGPT CLS CRC389 Baseline",
        "",
        "## Scope",
        "",
        "This evaluates the stage2 BiomeGPT raw CLS embeddings for the 389 CRC overlap samples, then applies a simple study mean-centering baseline before any GRL-corrected CLS experiment.",
        "",
        "The CLS matrix already existed at `outputs/crc_overlap_benchmark/biogpt_raw_cls_389.csv` and is recorded in the package reports as extracted from the stage2 checkpoint.",
        "",
        "## Important Metric Note",
        "",
        "The abundance methods and CLS methods are placed in the same Python MMUPHin-style diagnostic table. For CLS, R/Bray-Curtis abundance PERMANOVA is not directly applicable, so R2 is standardized Euclidean/linear partial R2 in representation space.",
        "",
        "## Primary Table",
        "",
        primary.to_markdown(index=False),
        "",
        "## Mean-Centering Effect",
        "",
        f"- Study BA changed from {raw_cls['study_prediction_balanced_accuracy']:.3f} to {centered['study_prediction_balanced_accuracy']:.3f}.",
        f"- Disease LOSO AUC changed from {raw_cls['disease_LOSO_mean_within_study_AUC']:.3f} to {centered['disease_LOSO_mean_within_study_AUC']:.3f}.",
        f"- Study R2 changed from {raw_cls['study_R2_condition_controlled']:.4f} to {centered['study_R2_condition_controlled']:.4f}.",
        f"- Condition R2 changed from {raw_cls['condition_R2_study_controlled']:.4f} to {centered['condition_R2_study_controlled']:.4f}.",
        "",
        "## Reading",
        "",
        "- This is a baseline diagnostic, not a GRL/scGPT result.",
        "- If mean-centering lowers study signal while preserving disease LOSO AUC, it is a useful simple baseline that GRL-corrected CLS must beat.",
        "- If mean-centering hurts disease LOSO AUC, GRL needs an explicit preservation mechanism.",
        "",
        "## Output Files",
        "",
    ]
    for key, value in paths.items():
        lines.append(f"- `{key}`: `{Path(value).relative_to(ROOT)}`")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "biogpt_cls_crc389_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(DATA_DIR / "metadata_389.csv", dtype=str)
    raw_cls_path = DATA_DIR / "biogpt_raw_cls_389.csv"
    if not raw_cls_path.exists():
        raise FileNotFoundError(f"Expected stage2 CLS matrix is missing: {raw_cls_path}")
    raw_cls = read_matrix(raw_cls_path)
    centered_cls = study_mean_center_cls(raw_cls, metadata)
    centered_path = DATA_DIR / "biogpt_mean_centered_cls_389.csv"
    centered_cls.to_csv(centered_path, index=False)

    rows: list[dict[str, Any]] = []
    for method, path, abundance_like, slug in method_specs():
        matrix = read_matrix(path)
        matrix = align_matrix(matrix, metadata)
        rows.extend(evaluate_method(method, matrix, metadata, abundance_like=abundance_like))
        save_pca_plot(matrix, metadata, "studyID", FIGURE_DIR / f"{slug}_pca_by_study.png")
        save_pca_plot(matrix, metadata, "study_condition", FIGURE_DIR / f"{slug}_pca_by_condition.png")
        print("evaluated", method)

    metrics = pd.DataFrame(rows)
    long_path = METRIC_DIR / "biogpt_cls_crc389_metrics_long.csv"
    primary_path = METRIC_DIR / "biogpt_cls_crc389_primary_table.csv"
    metrics.to_csv(long_path, index=False)
    primary = pivot_primary(metrics)
    order = {
        "Raw abundance": 0,
        "MMUPHin adjusted abundance": 1,
        "BiomeGPT raw CLS": 2,
        "BiomeGPT study-mean-centered CLS": 3,
    }
    primary["_order"] = primary["method"].map(order)
    primary = primary.sort_values("_order").drop(columns="_order")
    primary.to_csv(primary_path, index=False)
    paths = {
        "raw_cls": str(raw_cls_path),
        "mean_centered_cls": str(centered_path),
        "metrics_long": str(long_path),
        "primary_table": str(primary_path),
        "figure_dir": str(FIGURE_DIR),
    }
    write_summary(metrics, paths)
    print("BIOGPT_CLS_CRC389_BASELINE_OK")
    print(primary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
