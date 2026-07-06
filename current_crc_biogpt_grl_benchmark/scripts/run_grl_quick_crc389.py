from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.io import read_matrix
from src.evaluation.permanova import distance_r2
from src.evaluation.plots import save_pca_plot
from src.evaluation.probes import probe_classification
from src.grl_correction.train_grl import save_grl_result, train_embedding_grl


DATA_DIR = ROOT / "outputs" / "crc_overlap_benchmark"
METRIC_DIR = ROOT / "outputs" / "metrics"
FIGURE_DIR = ROOT / "outputs" / "figures" / "grl_quick_crc389"
REPORT_DIR = ROOT / "reports"


def evaluate_matrix(method: str, matrix: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, Any]:
    study = probe_classification(matrix, metadata, "studyID", random_state=42)
    condition = probe_classification(
        matrix,
        metadata,
        "study_condition",
        positive_label="CRC",
        random_state=42,
    )
    return {
        "method": method,
        "n_samples": int(len(matrix)),
        "n_features_or_dims": int(matrix.shape[1] - 1),
        "study_balanced_accuracy": study["balanced_accuracy"],
        "study_macro_f1": study["macro_f1"],
        "condition_auc": condition.get("auroc"),
        "condition_balanced_accuracy": condition["balanced_accuracy"],
        "condition_macro_f1": condition["macro_f1"],
        "study_permanova_r2": None,
        "condition_permanova_r2": None,
        "study_centroid_r2_fallback": distance_r2(matrix, metadata, "studyID"),
        "condition_centroid_r2_fallback": distance_r2(matrix, metadata, "study_condition"),
    }


def make_plots(method_slug: str, matrix: pd.DataFrame, metadata: pd.DataFrame) -> None:
    save_pca_plot(matrix, metadata, "studyID", FIGURE_DIR / f"{method_slug}_pca_by_study.png")
    save_pca_plot(matrix, metadata, "study_condition", FIGURE_DIR / f"{method_slug}_pca_by_condition.png")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_summary(comparison: pd.DataFrame, history: pd.DataFrame, paths: dict[str, str]) -> None:
    raw = comparison[comparison["method"] == "Raw abundance"].iloc[0]
    mmuphin = comparison[comparison["method"] == "MMUPHin adjusted abundance"].iloc[0]
    grl = comparison[comparison["method"] == "GRL corrected raw-abundance representation"].iloc[0]

    study_delta_raw = float(grl["study_balanced_accuracy"]) - float(raw["study_balanced_accuracy"])
    condition_delta_raw = float(grl["condition_auc"]) - float(raw["condition_auc"])
    study_vs_mmuphin = float(grl["study_balanced_accuracy"]) - float(mmuphin["study_balanced_accuracy"])
    condition_vs_mmuphin = float(grl["condition_auc"]) - float(mmuphin["condition_auc"])
    grl_reduced_study = study_delta_raw < 0
    cls_next = (
        "Not yet as a correction benchmark: this run increased study predictability. "
        "The next local step should be tuning the GRL objective on abundance features or running a clearly labeled CLS smoke check, not claiming improvement."
        if not grl_reduced_study
        else "Yes, cautiously: this run reduced study predictability while retaining CRC/control signal, so a raw BiomeGPT CLS smoke check is reasonable."
    )

    lines = [
        "# GRL Quick CRC389 Sanity Check",
        "",
        "## Scope",
        "",
        "This is a quick local abundance-level GRL prototype on the MMUPHin CRC 389-overlap benchmark. It does not run BiomeGPT, does not use A100, and is not a final scGPT/BiomeGPT result.",
        "",
        "## Inputs",
        "",
        "- `outputs/crc_overlap_benchmark/raw_abundance_389.csv`",
        "- `outputs/crc_overlap_benchmark/mmuphin_adjusted_abundance_389.csv`",
        "- `outputs/crc_overlap_benchmark/metadata_389.csv`",
        "",
        "## Methods Compared",
        "",
        "- Raw abundance",
        "- MMUPHin adjusted abundance",
        "- GRL corrected representation trained from raw abundance features",
        "",
        "## Metric Summary",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"- Did GRL reduce study predictability compared with raw abundance? {'Yes' if grl_reduced_study else 'No'}. Study balanced accuracy changed from {fmt(raw['study_balanced_accuracy'])} to {fmt(grl['study_balanced_accuracy'])}.",
        f"- Did GRL preserve CRC/control signal? {'Numerically yes' if condition_delta_raw > -0.03 else 'Not clearly'}. CRC/control AUROC changed from {fmt(raw['condition_auc'])} to {fmt(grl['condition_auc'])}. Because the GRL representation was trained using all condition labels, the near-perfect condition probe should be read as a sanity-check signal, not strict held-out disease generalization.",
        f"- Compared with MMUPHin, GRL study balanced accuracy was {'higher' if study_vs_mmuphin > 0 else 'lower'} by {fmt(abs(study_vs_mmuphin))}; CRC/control AUROC was {'higher' if condition_vs_mmuphin > 0 else 'lower'} by {fmt(abs(condition_vs_mmuphin))}.",
        f"- Is this promising enough to connect to BiomeGPT CLS next? {cls_next}",
        "",
        "## Training Notes",
        "",
        f"- Epochs run: {int(history['epoch'].max()) if not history.empty else 0}",
        f"- Final preservation loss: {fmt(history['preservation_loss'].iloc[-1]) if 'preservation_loss' in history and not history.empty else 'NA'}",
        f"- Final internal condition loss: {fmt(history['condition_loss'].iloc[-1]) if 'condition_loss' in history and not history.empty else 'NA'}",
        f"- Final internal study adversary loss: {fmt(history['study_adversary_loss'].iloc[-1]) if 'study_adversary_loss' in history and not history.empty else 'NA'}",
        "",
        "## Limitations",
        "",
        "- This is not BiomeGPT CLS yet.",
        "- This is not full scGPT-style training.",
        "- This uses abundance features as the input matrix.",
        "- Final comparison should use raw BiomeGPT CLS and GRL-corrected BiomeGPT CLS on the same 389 samples.",
        "- The external probes use cross-validation, but the learned GRL representation itself was fit on all 389 samples and their labels in this quick prototype.",
        "- External probe metrics matter more than internal GRL training loss.",
        "- `study_permanova_r2` and `condition_permanova_r2` are left blank because formal condition-controlled PERMANOVA is not implemented in this quick Python runner; centroid R2 fallback columns are provided only as local diagnostics.",
        "",
        "## Output Files",
        "",
    ]
    for key, value in paths.items():
        lines.append(f"- `{key}`: `{Path(value).relative_to(ROOT)}`")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "grl_quick_crc389_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = DATA_DIR / "metadata_389.csv"
    raw_path = DATA_DIR / "raw_abundance_389.csv"
    mmuphin_path = DATA_DIR / "mmuphin_adjusted_abundance_389.csv"
    for path in [metadata_path, raw_path, mmuphin_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = pd.read_csv(metadata_path, dtype=str)
    raw = read_matrix(raw_path)
    mmuphin = read_matrix(mmuphin_path)

    baseline_rows = [
        evaluate_matrix("Raw abundance", raw, metadata),
        evaluate_matrix("MMUPHin adjusted abundance", mmuphin, metadata),
    ]
    baseline = pd.DataFrame(baseline_rows)
    baseline_path = METRIC_DIR / "grl_quick_baseline_raw_vs_mmuphin.csv"
    baseline.to_csv(baseline_path, index=False)

    make_plots("raw_abundance", raw, metadata)
    make_plots("mmuphin_adjusted_abundance", mmuphin, metadata)

    result = train_embedding_grl(
        raw,
        metadata,
        latent_dim=min(64, raw.shape[1] - 1),
        epochs=100,
        lr=1e-3,
        lambda_grl=1.0,
        lambda_schedule="dann",
        warmup_fraction=0.4,
        condition_weight=1.0,
        preserve_weight=0.1,
        batch_size=64,
        seed=42,
        external_eval_every=25,
        device="cpu",
    )

    grl_output_dir = DATA_DIR / "grl_quick_raw_abundance"
    saved = save_grl_result(result, grl_output_dir)
    grl_output_path = DATA_DIR / "grl_corrected_raw_abundance_z_389.csv"
    result.corrected_embeddings.to_csv(grl_output_path, index=False)
    history_path = METRIC_DIR / "grl_training_history.csv"
    final_internal_path = METRIC_DIR / "grl_final_internal_metrics.csv"
    result.history.to_csv(history_path, index=False)
    result.final_probe_metrics.to_csv(final_internal_path, index=False)

    grl = read_matrix(grl_output_path)
    make_plots("grl_corrected_raw_abundance", grl, metadata)
    comparison = pd.concat(
        [
            baseline,
            pd.DataFrame([evaluate_matrix("GRL corrected raw-abundance representation", grl, metadata)]),
        ],
        ignore_index=True,
    )
    comparison_path = METRIC_DIR / "grl_quick_method_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    output_paths = {
        "baseline_metrics": str(baseline_path),
        "method_comparison": str(comparison_path),
        "grl_corrected_representation": str(grl_output_path),
        "grl_training_history": str(history_path),
        "grl_final_internal_metrics": str(final_internal_path),
        "figure_dir": str(FIGURE_DIR),
        "save_grl_result_dir": str(grl_output_dir),
    }
    (METRIC_DIR / "grl_quick_run_paths.json").write_text(json.dumps(output_paths, indent=2), encoding="utf-8")
    write_summary(comparison, result.history, output_paths)

    print("GRL_QUICK_CRC389_OK")
    print(comparison[["method", "study_balanced_accuracy", "condition_auc", "condition_balanced_accuracy"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
