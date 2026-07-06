from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.io import read_matrix
from src.evaluation.permanova import distance_r2
from src.evaluation.plots import save_pca_plot
from src.evaluation.probes import probe_classification


DEFAULT_METHODS = {
    "raw_abundance": ROOT / "outputs" / "crc_overlap_benchmark" / "raw_abundance_389.csv",
    "mmuphin_adjusted_abundance": ROOT / "outputs" / "crc_overlap_benchmark" / "mmuphin_adjusted_abundance_389.csv",
    "biogpt_raw_cls": ROOT / "outputs" / "crc_overlap_benchmark" / "biogpt_raw_cls_389.csv",
    "biogpt_grl_corrected_cls": ROOT / "outputs" / "crc_overlap_benchmark" / "biogpt_grl_corrected_cls_389.csv",
}


def evaluate_method(method_name: str, matrix_path: Path, metadata: pd.DataFrame) -> list[dict[str, object]]:
    matrix = read_matrix(matrix_path)
    rows = []
    study = probe_classification(matrix, metadata, "studyID")
    rows.append({"method": method_name, "metric": "study_prediction_balanced_accuracy", "value": study["balanced_accuracy"]})
    rows.append({"method": method_name, "metric": "study_prediction_macro_f1", "value": study["macro_f1"]})
    condition = probe_classification(matrix, metadata, "study_condition", positive_label="CRC")
    rows.append({"method": method_name, "metric": "condition_auroc", "value": condition.get("auroc")})
    rows.append({"method": method_name, "metric": "condition_balanced_accuracy", "value": condition["balanced_accuracy"]})
    rows.append({"method": method_name, "metric": "condition_macro_f1", "value": condition["macro_f1"]})
    rows.append({"method": method_name, "metric": "study_centroid_r2_fallback", "value": distance_r2(matrix, metadata, "studyID")})
    rows.append({"method": method_name, "metric": "condition_centroid_r2_fallback", "value": distance_r2(matrix, metadata, "study_condition")})
    save_pca_plot(matrix, metadata, "studyID", ROOT / "outputs" / "figures" / f"{method_name}_pca_by_study.png")
    save_pca_plot(matrix, metadata, "study_condition", ROOT / "outputs" / "figures" / f"{method_name}_pca_by_condition.png")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(ROOT / "outputs" / "crc_overlap_benchmark" / "metadata_389.csv"))
    parser.add_argument("--method", action="append", help="name=path. May be repeated.")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata, dtype=str)
    methods = DEFAULT_METHODS.copy()
    if args.method:
        methods = {}
        for item in args.method:
            name, path = item.split("=", 1)
            methods[name] = Path(path)

    rows = []
    skipped = []
    for method_name, path in methods.items():
        if not Path(path).exists():
            skipped.append({"method": method_name, "path": str(path), "reason": "missing"})
            continue
        try:
            rows.extend(evaluate_method(method_name, Path(path), metadata))
        except Exception as exc:
            skipped.append({"method": method_name, "path": str(path), "reason": f"{type(exc).__name__}: {exc}"})

    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "crc_method_comparison_metrics.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(out_dir / "crc_method_comparison_skipped.csv", index=False)
        lines = ["# CRC Method Evaluation Skipped Items", "", "| method | path | reason |", "|---|---|---|"]
        for row in skipped:
            lines.append(f"| {row['method']} | {row['path']} | {row['reason']} |")
        (ROOT / "reports" / "evaluation_skipped_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"evaluated_methods={len(set(r['method'] for r in rows))}")
    print(f"skipped_methods={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
