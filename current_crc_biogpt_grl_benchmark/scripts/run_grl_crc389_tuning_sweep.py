from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grl_quick_crc389 import evaluate_matrix, fmt, make_plots
from src.evaluation.io import read_matrix
from src.grl_correction.train_grl import save_grl_result, train_embedding_grl


DATA_DIR = ROOT / "outputs" / "crc_overlap_benchmark"
METRIC_DIR = ROOT / "outputs" / "metrics"
REPORT_DIR = ROOT / "reports"
SWEEP_DIR = DATA_DIR / "grl_tuning_sweep_crc389"
FIGURE_DIR = ROOT / "outputs" / "figures" / "grl_tuning_crc389"


CASES: list[dict[str, Any]] = [
    {
        "case_id": "A_condaware_cw05_pw001_lam1",
        "latent_dim": 64,
        "condition_aware_adversary": True,
        "condition_weight": 0.5,
        "preserve_weight": 0.01,
        "lambda_grl": 1.0,
        "lambda_schedule": "linear",
        "warmup_fraction": 0.1,
        "use_study_conditioned_decoder": False,
    },
    {
        "case_id": "B_condaware_cw02_pw001_lam2",
        "latent_dim": 64,
        "condition_aware_adversary": True,
        "condition_weight": 0.2,
        "preserve_weight": 0.01,
        "lambda_grl": 2.0,
        "lambda_schedule": "linear",
        "warmup_fraction": 0.1,
        "use_study_conditioned_decoder": False,
    },
    {
        "case_id": "C_condaware_cw05_pw000_lam2",
        "latent_dim": 64,
        "condition_aware_adversary": True,
        "condition_weight": 0.5,
        "preserve_weight": 0.0,
        "lambda_grl": 2.0,
        "lambda_schedule": "linear",
        "warmup_fraction": 0.1,
        "use_study_conditioned_decoder": False,
    },
    {
        "case_id": "D_condaware_cw05_pw001_lam5_studydecoder",
        "latent_dim": 64,
        "condition_aware_adversary": True,
        "condition_weight": 0.5,
        "preserve_weight": 0.01,
        "lambda_grl": 5.0,
        "lambda_schedule": "linear",
        "warmup_fraction": 0.1,
        "use_study_conditioned_decoder": True,
    },
    {
        "case_id": "E_bottleneck16_no_condition_no_preserve_lam5",
        "latent_dim": 16,
        "condition_aware_adversary": True,
        "condition_weight": 0.0,
        "preserve_weight": 0.0,
        "lambda_grl": 5.0,
        "lambda_schedule": "linear",
        "warmup_fraction": 0.1,
        "use_study_conditioned_decoder": False,
    },
    {
        "case_id": "F_bottleneck16_cw01_no_preserve_lam10",
        "latent_dim": 16,
        "condition_aware_adversary": True,
        "condition_weight": 0.1,
        "preserve_weight": 0.0,
        "lambda_grl": 10.0,
        "lambda_schedule": "linear",
        "warmup_fraction": 0.1,
        "use_study_conditioned_decoder": False,
    },
    {
        "case_id": "G_bottleneck8_cw01_pw0001_lam10",
        "latent_dim": 8,
        "condition_aware_adversary": True,
        "condition_weight": 0.1,
        "preserve_weight": 0.001,
        "lambda_grl": 10.0,
        "lambda_schedule": "linear",
        "warmup_fraction": 0.1,
        "use_study_conditioned_decoder": False,
    },
    {
        "case_id": "H_bottleneck8_no_condition_no_preserve_lam10",
        "latent_dim": 8,
        "condition_aware_adversary": True,
        "condition_weight": 0.0,
        "preserve_weight": 0.0,
        "lambda_grl": 10.0,
        "lambda_schedule": "linear",
        "warmup_fraction": 0.1,
        "use_study_conditioned_decoder": False,
    },
]


def final_probe_dict(probes: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for _, row in probes.iterrows():
        out[str(row["metric"])] = row.get("value")
    return out


def choose_best(sweep: pd.DataFrame, raw_study: float, raw_auc: float) -> pd.Series:
    scored = sweep.copy()
    scored["meets_study_target"] = scored["study_balanced_accuracy"] < raw_study
    scored["meets_condition_floor"] = scored["condition_auc"] >= raw_auc - 0.05
    valid = scored[scored["meets_study_target"] & scored["meets_condition_floor"]]
    if not valid.empty:
        return valid.sort_values(["study_balanced_accuracy", "condition_auc"], ascending=[True, False]).iloc[0]
    return scored.sort_values(["study_balanced_accuracy", "condition_auc"], ascending=[True, False]).iloc[0]


def write_summary(
    baseline: pd.DataFrame,
    sweep: pd.DataFrame,
    best: pd.Series,
    output_paths: dict[str, str],
) -> None:
    raw = baseline[baseline["method"] == "Raw abundance"].iloc[0]
    mmuphin = baseline[baseline["method"] == "MMUPHin adjusted abundance"].iloc[0]
    reduced_vs_raw = float(best["study_balanced_accuracy"]) < float(raw["study_balanced_accuracy"])
    preserved_vs_raw = float(best["condition_auc"]) >= float(raw["condition_auc"]) - 0.05

    lines = [
        "# GRL CRC389 Tuning Sweep",
        "",
        "## Scope",
        "",
        "This is a small local tuning sweep for the abundance-level GRL prototype on the MMUPHin CRC 389-overlap benchmark. It does not run BiomeGPT, does not use A100, and is not a final scGPT/BiomeGPT result.",
        "",
        "## Why This Sweep",
        "",
        "The first quick GRL run preserved CRC/control too strongly and increased external study predictability. This sweep tests condition-aware study adversaries, weaker condition preservation, weaker preservation loss, stronger GRL pressure, and a study-conditioned decoder.",
        "",
        "## Baseline Reference",
        "",
        baseline.to_markdown(index=False),
        "",
        "## Sweep Results",
        "",
        sweep.to_markdown(index=False),
        "",
        "## Best Local Setting",
        "",
        f"- Selected case: `{best['case_id']}`",
        f"- Study balanced accuracy: {fmt(best['study_balanced_accuracy'])} versus raw {fmt(raw['study_balanced_accuracy'])} and MMUPHin {fmt(mmuphin['study_balanced_accuracy'])}.",
        f"- CRC/control AUROC: {fmt(best['condition_auc'])} versus raw {fmt(raw['condition_auc'])} and MMUPHin {fmt(mmuphin['condition_auc'])}.",
        f"- Did it reduce study predictability versus raw? {'Yes' if reduced_vs_raw else 'No'}.",
        f"- Did it keep CRC/control AUROC within 0.05 of raw? {'Yes' if preserved_vs_raw else 'No'}.",
        "",
        "## Interpretation",
        "",
        "- Treat this as local objective debugging only. The representation was trained on all 389 samples with labels, and external probes are cross-validated on the learned representation.",
        "- If none of the cases reduce study predictability below raw abundance, the current GRL objective is still not a controlled correction method for this abundance benchmark.",
        "- If a case reduces study predictability while preserving condition signal, it is a candidate scaffold for a clearly labeled BiomeGPT CLS smoke check, not a final result.",
        "",
        "## Output Files",
        "",
    ]
    for key, value in output_paths.items():
        lines.append(f"- `{key}`: `{Path(value).relative_to(ROOT)}`")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "grl_crc389_tuning_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(DATA_DIR / "metadata_389.csv", dtype=str)
    raw = read_matrix(DATA_DIR / "raw_abundance_389.csv")
    mmuphin = read_matrix(DATA_DIR / "mmuphin_adjusted_abundance_389.csv")
    baseline = pd.DataFrame(
        [
            evaluate_matrix("Raw abundance", raw, metadata),
            evaluate_matrix("MMUPHin adjusted abundance", mmuphin, metadata),
        ]
    )

    rows: list[dict[str, Any]] = []
    for case in CASES:
        case_id = str(case["case_id"])
        params = {k: v for k, v in case.items() if k != "case_id"}
        result = train_embedding_grl(
            raw,
            metadata,
            latent_dim=min(int(params.pop("latent_dim")), raw.shape[1] - 1),
            epochs=100,
            lr=1e-3,
            batch_size=64,
            seed=42,
            external_eval_every=50,
            device="cpu",
            **params,
        )
        case_dir = SWEEP_DIR / case_id
        save_grl_result(result, case_dir)
        corrected_path = case_dir / "grl_corrected_raw_abundance_z_389.csv"
        history_path = case_dir / "grl_training_history.csv"
        probes_path = case_dir / "grl_final_external_probe_metrics.csv"
        result.corrected_embeddings.to_csv(corrected_path, index=False)
        result.history.to_csv(history_path, index=False)
        result.final_probe_metrics.to_csv(probes_path, index=False)

        eval_row = evaluate_matrix(f"GRL {case_id}", result.corrected_embeddings, metadata)
        probe_row = final_probe_dict(result.final_probe_metrics)
        rows.append(
            {
                "case_id": case_id,
                "lambda_grl": params["lambda_grl"],
                "latent_dim": result.config["latent_dim"],
                "lambda_schedule": params["lambda_schedule"],
                "warmup_fraction": params["warmup_fraction"],
                "condition_weight": params["condition_weight"],
                "preserve_weight": params["preserve_weight"],
                "condition_aware_adversary": params["condition_aware_adversary"],
                "use_study_conditioned_decoder": params["use_study_conditioned_decoder"],
                "n_samples": eval_row["n_samples"],
                "n_features_or_dims": eval_row["n_features_or_dims"],
                "study_balanced_accuracy": eval_row["study_balanced_accuracy"],
                "study_macro_f1": eval_row["study_macro_f1"],
                "condition_auc": eval_row["condition_auc"],
                "condition_balanced_accuracy": eval_row["condition_balanced_accuracy"],
                "condition_macro_f1": eval_row["condition_macro_f1"],
                "study_centroid_r2_fallback": eval_row["study_centroid_r2_fallback"],
                "condition_centroid_r2_fallback": eval_row["condition_centroid_r2_fallback"],
                "final_internal_study_balanced_accuracy": probe_row.get("study_balanced_accuracy"),
                "final_internal_condition_auroc": probe_row.get("condition_auroc"),
                "corrected_path": str(corrected_path),
            }
        )
        print(
            case_id,
            "study_bacc=",
            fmt(eval_row["study_balanced_accuracy"]),
            "cond_auc=",
            fmt(eval_row["condition_auc"]),
        )

    sweep = pd.DataFrame(rows)
    raw_row = baseline[baseline["method"] == "Raw abundance"].iloc[0]
    best = choose_best(sweep, float(raw_row["study_balanced_accuracy"]), float(raw_row["condition_auc"]))
    best_path = DATA_DIR / "grl_tuned_best_raw_abundance_z_389.csv"
    pd.read_csv(best["corrected_path"]).to_csv(best_path, index=False)

    make_plots("grl_tuned_best_raw_abundance", read_matrix(best_path), metadata)
    baseline_path = METRIC_DIR / "grl_crc389_tuning_baseline.csv"
    sweep_path = METRIC_DIR / "grl_crc389_tuning_sweep.csv"
    best_path_json = METRIC_DIR / "grl_crc389_tuning_best.json"
    baseline.to_csv(baseline_path, index=False)
    sweep.to_csv(sweep_path, index=False)
    best_payload = best.drop(labels=["corrected_path"]).to_dict()
    best_payload["corrected_path"] = str(best_path)
    best_path_json.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    output_paths = {
        "baseline_metrics": str(baseline_path),
        "sweep_metrics": str(sweep_path),
        "best_config": str(best_path_json),
        "best_corrected_representation": str(best_path),
        "sweep_result_dir": str(SWEEP_DIR),
        "best_figure_dir": str(FIGURE_DIR),
    }
    write_summary(baseline, sweep.drop(columns=["corrected_path"]), best, output_paths)

    print("GRL_CRC389_TUNING_SWEEP_OK")
    print(sweep[["case_id", "study_balanced_accuracy", "condition_auc", "condition_balanced_accuracy"]].to_string(index=False))
    print("BEST", best["case_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
