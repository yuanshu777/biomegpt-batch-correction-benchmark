from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grl_quick_crc389 import evaluate_matrix, fmt
from src.evaluation.io import read_matrix
from src.evaluation.plots import save_pca_plot
from src.grl_correction.crossfit import (
    EarlyStoppingConfig,
    align_matrix_to_metadata,
    split_train_validation,
    stratified_sample_folds,
    train_grl_with_validation_early_stopping,
)


DATA_DIR = ROOT / "outputs" / "crc_overlap_benchmark"
METRIC_DIR = ROOT / "outputs" / "metrics"
FIGURE_DIR = ROOT / "outputs" / "figures" / "grl_crossfit_crc389"
REPORT_DIR = ROOT / "reports"

PRIMARY_SEED = 42
CROSSFIT_SEEDS = [42, 99, 2026]
N_SPLITS = 5
VALIDATION_FRACTION = 0.2


def sample_ids(metadata: pd.DataFrame) -> set[str]:
    return set(metadata["sample_id"].astype(str).tolist())


def subset_matrix(matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    ids = sample_ids(metadata)
    return matrix[matrix["sample_id"].astype(str).isin(ids)].copy()


def add_tradeoff_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    raw = out[out["method"] == "Raw abundance"].iloc[0]
    raw_study = float(raw["study_balanced_accuracy"])
    raw_auc = float(raw["condition_auc"])
    out["batch_reduction"] = raw_study - out["study_balanced_accuracy"].astype(float)
    out["biology_change"] = out["condition_auc"].astype(float) - raw_auc
    out["biology_retention"] = out["condition_auc"].astype(float) / raw_auc if raw_auc else pd.NA
    out["useful_by_rule"] = (out["batch_reduction"] > 0) & (out["biology_change"] >= -0.05)
    return out


def evaluate_and_plot(method: str, slug: str, matrix: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, Any]:
    save_pca_plot(matrix, metadata, "studyID", FIGURE_DIR / f"{slug}_pca_by_study.png")
    save_pca_plot(matrix, metadata, "study_condition", FIGURE_DIR / f"{slug}_pca_by_condition.png")
    return evaluate_matrix(method, matrix, metadata)


def selected_to_fold_metrics(
    selected: dict[str, Any],
    heldout_eval: dict[str, Any],
    seed: int,
    outer_fold: int,
    n_outer_train: int,
    n_inner_train: int,
    n_selection_val: int,
    n_heldout: int,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "outer_fold": outer_fold,
        "n_outer_train": n_outer_train,
        "n_inner_train": n_inner_train,
        "n_selection_val": n_selection_val,
        "n_heldout": n_heldout,
        "selected_epoch": selected.get("epoch"),
        "selection_reason": selected.get("selection_reason"),
        "constraint_satisfied": selected.get("constraint_satisfied"),
        "raw_validation_condition_auc": selected.get("raw_validation_condition_auc"),
        "validation_study_balanced_accuracy": selected.get("validation_study_balanced_accuracy"),
        "validation_condition_auc": selected.get("validation_condition_auc"),
        "validation_condition_floor": selected.get("condition_floor"),
        "validation_condition_shortfall": selected.get("condition_shortfall"),
        "validation_tradeoff_score": selected.get("tradeoff_score"),
        "heldout_study_balanced_accuracy": heldout_eval["study_balanced_accuracy"],
        "heldout_study_macro_f1": heldout_eval["study_macro_f1"],
        "heldout_condition_auc": heldout_eval["condition_auc"],
        "heldout_condition_balanced_accuracy": heldout_eval["condition_balanced_accuracy"],
        "heldout_condition_macro_f1": heldout_eval["condition_macro_f1"],
        "heldout_study_centroid_r2_fallback": heldout_eval["study_centroid_r2_fallback"],
        "heldout_condition_centroid_r2_fallback": heldout_eval["condition_centroid_r2_fallback"],
    }


def make_config(seed: int) -> EarlyStoppingConfig:
    return EarlyStoppingConfig(
        latent_dim=8,
        epochs=100,
        lr=1e-3,
        lambda_grl=10.0,
        lambda_schedule="linear",
        warmup_fraction=0.1,
        condition_weight=0.1,
        preserve_weight=0.001,
        condition_aware_adversary=True,
        use_study_conditioned_decoder=False,
        eval_every=5,
        condition_auc_margin=0.05,
        patience_evals=6,
        seed=seed,
        device="cpu",
    )


def run_full_data_tuned(raw: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = make_config(PRIMARY_SEED)
    train_meta, val_meta = split_train_validation(metadata, VALIDATION_FRACTION, PRIMARY_SEED)
    result = train_grl_with_validation_early_stopping(
        subset_matrix(raw, train_meta),
        train_meta,
        subset_matrix(raw, val_meta),
        val_meta,
        raw,
        config=config,
        fold_id="full_data_seed42",
    )
    trace = result.trace.copy()
    trace["seed"] = PRIMARY_SEED
    trace["outer_fold"] = "full_data"
    trace["stage"] = "full_data_tuned"
    return align_matrix_to_metadata(result.corrected_embeddings, metadata), trace, result.selected


def run_crossfit(raw: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_fold_rows: list[dict[str, Any]] = []
    all_trace_rows: list[pd.DataFrame] = []
    primary_outputs: list[pd.DataFrame] = []

    for seed in CROSSFIT_SEEDS:
        folds = stratified_sample_folds(metadata, N_SPLITS, seed)
        for fold_number, (outer_train_idx, heldout_idx) in enumerate(folds, start=1):
            outer_train_meta = metadata.iloc[outer_train_idx].copy()
            heldout_meta = metadata.iloc[heldout_idx].copy()
            inner_train_meta, selection_val_meta = split_train_validation(
                outer_train_meta,
                VALIDATION_FRACTION,
                seed + fold_number,
            )
            config = make_config(seed)
            result = train_grl_with_validation_early_stopping(
                subset_matrix(raw, inner_train_meta),
                inner_train_meta,
                subset_matrix(raw, selection_val_meta),
                selection_val_meta,
                subset_matrix(raw, heldout_meta),
                config=config,
                fold_id=f"seed{seed}_fold{fold_number}",
            )
            heldout_z = align_matrix_to_metadata(result.corrected_embeddings, heldout_meta)
            heldout_eval = evaluate_matrix(
                f"Cross-fit heldout seed {seed} fold {fold_number}",
                heldout_z,
                heldout_meta,
            )
            all_fold_rows.append(
                selected_to_fold_metrics(
                    result.selected,
                    heldout_eval,
                    seed,
                    fold_number,
                    n_outer_train=len(outer_train_meta),
                    n_inner_train=len(inner_train_meta),
                    n_selection_val=len(selection_val_meta),
                    n_heldout=len(heldout_meta),
                )
            )
            trace = result.trace.copy()
            trace["seed"] = seed
            trace["outer_fold"] = fold_number
            trace["stage"] = "crossfit"
            all_trace_rows.append(trace)
            if seed == PRIMARY_SEED:
                primary_outputs.append(heldout_z)
            print(
                f"seed={seed} fold={fold_number}",
                "selected_epoch=",
                result.selected.get("epoch"),
                "heldout_study_ba=",
                fmt(heldout_eval["study_balanced_accuracy"]),
                "heldout_condition_auc=",
                fmt(heldout_eval["condition_auc"]),
            )

    primary_oof = pd.concat(primary_outputs, ignore_index=True)
    primary_oof = align_matrix_to_metadata(primary_oof, metadata)
    per_fold = pd.DataFrame(all_fold_rows)
    trace = pd.concat(all_trace_rows, ignore_index=True)
    return primary_oof, per_fold, trace


def write_summary(
    comparison: pd.DataFrame,
    per_fold: pd.DataFrame,
    full_selected: dict[str, Any],
    output_paths: dict[str, str],
) -> None:
    raw = comparison[comparison["method"] == "Raw abundance"].iloc[0]
    mmuphin = comparison[comparison["method"] == "MMUPHin adjusted abundance"].iloc[0]
    full = comparison[comparison["method"] == "Full-data tuned GRL"].iloc[0]
    crossfit = comparison[comparison["method"] == "Cross-fitted tuned GRL"].iloc[0]
    fold_summary = per_fold[
        ["heldout_study_balanced_accuracy", "heldout_condition_auc", "heldout_condition_balanced_accuracy"]
    ].agg(["mean", "std", "min", "max"])
    seed_summary = (
        per_fold.groupby("seed")[["heldout_study_balanced_accuracy", "heldout_condition_auc"]]
        .mean()
        .reset_index()
    )
    constraint_rate = float(per_fold["constraint_satisfied"].astype(bool).mean())

    crossfit_preserves_raw = bool(crossfit["biology_change"] >= -0.05)
    crossfit_reduces_batch = bool(crossfit["batch_reduction"] > 0)
    if float(crossfit["study_balanced_accuracy"]) < float(mmuphin["study_balanced_accuracy"]) and float(crossfit["condition_auc"]) >= float(mmuphin["condition_auc"]):
        verdict = "Cross-fitted GRL is stronger than MMUPHin on these probe metrics, but this remains an abundance-level prototype."
    elif crossfit_reduces_batch and crossfit_preserves_raw:
        verdict = "Cross-fitted GRL beats raw on study predictability, but it does not clearly beat MMUPHin."
    elif crossfit_reduces_batch:
        verdict = "Cross-fitted GRL reduces study predictability versus raw, but it fails the disease-signal preservation rule and is not yet useful by the predefined criterion."
    else:
        verdict = "Cross-fitted GRL does not yet beat raw on study predictability."

    lines = [
        "# GRL Cross-Fit CRC389 Summary",
        "",
        "## Scope",
        "",
        "This is a stricter local abundance-level GRL evaluation on the MMUPHin CRC 389-overlap benchmark. It does not run BiomeGPT and is not a final scGPT/BiomeGPT result.",
        "",
        "## Methods",
        "",
        "- Raw abundance",
        "- MMUPHin adjusted abundance",
        "- Full-data tuned GRL: validation-selected but not out-of-fold for every sample",
        "- Cross-fitted tuned GRL: each sample's corrected representation is generated by a model that did not train on that sample",
        "",
        "## Selection Rule",
        "",
        "- A checkpoint is eligible when validation condition AUROC is at least raw validation AUROC minus 0.05.",
        "- Among eligible checkpoints, choose the checkpoint with lowest validation study balanced accuracy.",
        "- If no checkpoint is eligible, choose the best penalized tradeoff and mark the fold as a constraint failure.",
        "",
        "## Method Comparison",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Cross-Fit Fold/Seed Stability",
        "",
        fold_summary.to_markdown(),
        "",
        "## Cross-Fit Seed Means",
        "",
        seed_summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"- Full-data tuned GRL selected epoch {int(full_selected['epoch'])}; this is not out-of-fold and should not be used for final claims.",
        f"- Cross-fit checkpoint condition constraint satisfaction rate: {constraint_rate:.2%}.",
        "- The validation condition constraint did not reliably transfer to held-out condition AUROC; this is why the cross-fitted method fails the useful-by-rule flag.",
        f"- Main verdict: {verdict}",
        "- Do not claim GRL beats MMUPHin based on a single seed.",
        "- The cross-fitted result is the relevant robustness result; the full-data result is only a transductive diagnostic.",
        "",
        "## Output Files",
        "",
    ]
    for key, value in output_paths.items():
        lines.append(f"- `{key}`: `{Path(value).relative_to(ROOT)}`")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "grl_crossfit_crc389_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(DATA_DIR / "metadata_389.csv", dtype=str)
    raw = read_matrix(DATA_DIR / "raw_abundance_389.csv")
    mmuphin = read_matrix(DATA_DIR / "mmuphin_adjusted_abundance_389.csv")

    full_z, full_trace, full_selected = run_full_data_tuned(raw, metadata)
    full_path = DATA_DIR / "grl_full_data_tuned_z_389.csv"
    full_z.to_csv(full_path, index=False)

    crossfit_z, per_fold, crossfit_trace = run_crossfit(raw, metadata)
    crossfit_path = DATA_DIR / "grl_crossfit_corrected_z_389.csv"
    crossfit_z.to_csv(crossfit_path, index=False)

    method_rows = [
        evaluate_and_plot("Raw abundance", "raw_abundance", raw, metadata),
        evaluate_and_plot("MMUPHin adjusted abundance", "mmuphin_adjusted_abundance", mmuphin, metadata),
        evaluate_and_plot("Full-data tuned GRL", "full_data_tuned_grl", full_z, metadata),
        evaluate_and_plot("Cross-fitted tuned GRL", "crossfitted_tuned_grl", crossfit_z, metadata),
    ]
    comparison = add_tradeoff_metrics(pd.DataFrame(method_rows))

    comparison_path = METRIC_DIR / "grl_crossfit_method_comparison.csv"
    per_fold_path = METRIC_DIR / "grl_crossfit_per_fold_metrics.csv"
    trace_path = METRIC_DIR / "grl_crossfit_early_stopping_trace.csv"
    selected_path = METRIC_DIR / "grl_crossfit_full_data_selected.json"
    comparison.to_csv(comparison_path, index=False)
    per_fold.to_csv(per_fold_path, index=False)
    pd.concat([full_trace, crossfit_trace], ignore_index=True).to_csv(trace_path, index=False)
    selected_path.write_text(json.dumps(full_selected, indent=2), encoding="utf-8")

    output_paths = {
        "method_comparison": str(comparison_path),
        "per_fold_metrics": str(per_fold_path),
        "early_stopping_trace": str(trace_path),
        "crossfit_corrected_representation": str(crossfit_path),
        "full_data_tuned_representation": str(full_path),
        "full_data_selected_checkpoint": str(selected_path),
        "figure_dir": str(FIGURE_DIR),
    }
    write_summary(comparison, per_fold, full_selected, output_paths)
    print("GRL_CROSSFIT_CRC389_OK")
    print(comparison[["method", "study_balanced_accuracy", "condition_auc", "batch_reduction", "biology_change", "useful_by_rule"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
