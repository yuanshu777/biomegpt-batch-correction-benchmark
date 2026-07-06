"""
Representation-level batch-correction baselines using real study ids.

This script is deliberately separate from model fine-tuning. It asks a simpler
diagnostic question: if we use the uploaded real study ids directly, how much of
the study signal in the phase-2 sample embeddings can be removed by a transparent
mean-only correction, and what happens to phenotype / H-D probes?

The correction is cross-fitted by study label so a sample is not corrected using
its own embedding in the study mean. This makes the probe results less circular
than fitting one set of batch means on all samples and evaluating on the same
points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from batch_effect_diagnostics import (
    keep_labels_with_min_count,
    make_hd_labels,
    multiclass_probe,
    str_to_bool_series,
)


def select_panel(meta: pd.DataFrame, panel: str) -> pd.Series:
    if panel == "high_only":
        return meta["external_confidence"].astype(str).str.lower().eq("high")
    if panel == "high_medium":
        return meta["external_confidence"].astype(str).str.lower().isin(["high", "medium"])
    if panel == "conservative_safe":
        return str_to_bool_series(meta["safe_for_final_batch_correction_conservative"])
    raise ValueError(f"Unknown panel: {panel}")


def crossfit_study_center(
    x: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
    seed: int,
    method: str,
    eps: float,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Apply study-wise mean or mean+scale correction in held-out folds."""
    unique, y = np.unique(labels, return_inverse=True)
    counts = np.bincount(y)
    actual_splits = int(min(n_splits, counts.min()))
    if actual_splits < 2:
        raise ValueError("Need at least two samples per study for cross-fitted correction")

    corrected = np.zeros_like(x, dtype=np.float32)
    fold_summaries: List[Dict[str, object]] = []
    splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)

    for fold, (fit_idx, holdout_idx) in enumerate(splitter.split(x, y), start=1):
        fit_x = x[fit_idx]
        fit_y = y[fit_idx]
        global_mean = fit_x.mean(axis=0)
        global_std = fit_x.std(axis=0) + eps

        batch_mean: Dict[int, np.ndarray] = {}
        batch_std: Dict[int, np.ndarray] = {}
        for cls in np.unique(fit_y):
            cls_x = fit_x[fit_y == cls]
            batch_mean[int(cls)] = cls_x.mean(axis=0)
            batch_std[int(cls)] = cls_x.std(axis=0) + eps

        holdout_y = y[holdout_idx]
        holdout_x = x[holdout_idx]
        holdout_corr = np.empty_like(holdout_x, dtype=np.float32)
        for row_i, cls in enumerate(holdout_y):
            if method == "mean_center":
                holdout_corr[row_i] = holdout_x[row_i] - batch_mean[int(cls)] + global_mean
            elif method == "mean_scale":
                holdout_corr[row_i] = (
                    (holdout_x[row_i] - batch_mean[int(cls)])
                    / batch_std[int(cls)]
                    * global_std
                    + global_mean
                )
            else:
                raise ValueError(f"Unknown method: {method}")
        corrected[holdout_idx] = holdout_corr

        fold_summaries.append(
            {
                "fold": fold,
                "n_fit": int(len(fit_idx)),
                "n_holdout": int(len(holdout_idx)),
                "n_studies": int(len(unique)),
                "method": method,
            }
        )

    return corrected, fold_summaries


def evaluate_panel(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    panel: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, object]:
    panel_mask = select_panel(meta, panel)
    panel_meta = meta.loc[panel_mask].copy()
    panel_x = embeddings[panel_mask.to_numpy()]
    batch_labels = keep_labels_with_min_count(
        panel_meta["batch_label_external_recommended"].astype(str),
        args.min_batch_size,
    )
    valid = batch_labels.notna()
    panel_meta = panel_meta.loc[valid].copy()
    panel_x = panel_x[valid.to_numpy()]
    batch_labels = batch_labels.loc[valid].astype(str)

    corrected, fold_summaries = crossfit_study_center(
        panel_x.astype(np.float32),
        batch_labels.to_numpy(),
        args.n_splits,
        args.seed,
        args.method,
        args.eps,
    )
    np.savez_compressed(
        out_dir / f"real_study_{panel}_{args.method}_corrected_embeddings.npz",
        embeddings=corrected.astype(np.float32),
        sample_ids=panel_meta.index.astype(str).to_numpy(),
        batch_labels=batch_labels.to_numpy(),
        method=np.array([args.method]),
        panel=np.array([panel]),
    )

    phenotype_col = "Phenotype_fullname" if "Phenotype_fullname" in panel_meta.columns else "Phenotype"
    panel_meta["hd_label"] = make_hd_labels(panel_meta[phenotype_col])

    probes = [
        multiclass_probe(panel_x, batch_labels, f"{panel}_study_before", args.seed, args.test_size),
        multiclass_probe(corrected, batch_labels, f"{panel}_study_after_{args.method}", args.seed, args.test_size),
        multiclass_probe(panel_x, panel_meta[phenotype_col].astype(str), f"{panel}_phenotype_before", args.seed, args.test_size),
        multiclass_probe(
            corrected,
            panel_meta[phenotype_col].astype(str),
            f"{panel}_phenotype_after_{args.method}",
            args.seed,
            args.test_size,
        ),
        multiclass_probe(panel_x, panel_meta["hd_label"], f"{panel}_hd_before", args.seed, args.test_size),
        multiclass_probe(corrected, panel_meta["hd_label"], f"{panel}_hd_after_{args.method}", args.seed, args.test_size),
    ]

    return {
        "panel": panel,
        "method": args.method,
        "n_samples": int(len(panel_meta)),
        "n_studies": int(batch_labels.nunique()),
        "min_study_count": int(batch_labels.value_counts().min()),
        "max_study_count": int(batch_labels.value_counts().max()),
        "folds": fold_summaries,
        "probe_results": probes,
    }


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(args.embeddings_npz, allow_pickle=True)
    embeddings = payload["embeddings"].astype(np.float32)
    sample_ids = payload["sample_ids"].astype(str)
    meta = pd.read_csv(args.batch_annotation_csv, low_memory=False).set_index("sample_id")
    meta = meta.loc[sample_ids].copy()

    panel_results = [evaluate_panel(embeddings, meta, panel, args, out_dir) for panel in args.panels]
    probes: List[Dict[str, object]] = []
    for result in panel_results:
        probes.extend(result["probe_results"])

    summary = {
        "run_config": vars(args),
        "embedding_shape": list(embeddings.shape),
        "n_samples_with_metadata": int(len(meta)),
        "panel_results": panel_results,
        "interpretation": {
            "batch_probe_rule": "A useful correction lowers study macro-F1 / balanced accuracy.",
            "biology_probe_rule": "H-D and phenotype probes are sanity checks; collapse means the correction removed biological signal too.",
            "method_note": (
                "mean_center subtracts the cross-fitted study mean and adds the global mean. "
                "mean_scale additionally maps each study variance to the global variance."
            ),
        },
    }

    with open(out_dir / f"real_study_embedding_correction_{args.method}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(probes).to_csv(out_dir / f"real_study_embedding_correction_{args.method}_probe_metrics.csv", index=False)

    compact_rows = []
    for result in panel_results:
        for probe in result["probe_results"]:
            compact_rows.append(
                {
                    "panel": result["panel"],
                    "method": result["method"],
                    "label_name": probe.get("label_name"),
                    "status": probe.get("status"),
                    "n_samples": probe.get("n_samples"),
                    "n_classes": probe.get("n_classes"),
                    "balanced_accuracy": probe.get("balanced_accuracy"),
                    "macro_f1": probe.get("macro_f1"),
                    "accuracy": probe.get("accuracy"),
                }
            )
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(out_dir / f"real_study_embedding_correction_{args.method}_compact_metrics.csv", index=False)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-study-id embedding correction baselines")
    p.add_argument("--embeddings_npz", default="dataset_v3/outputs_batch_diagnostics_real_study/phase2_sample_prompt_embeddings.npz")
    p.add_argument("--batch_annotation_csv", default="dataset_v3/meta_pretraining_phase2_gut_real_study_annotation.csv")
    p.add_argument("--output_dir", default="dataset_v3/outputs_real_study_embedding_correction")
    p.add_argument("--panels", nargs="+", default=["high_only", "conservative_safe"])
    p.add_argument("--method", choices=["mean_center", "mean_scale"], default="mean_center")
    p.add_argument("--min_batch_size", type=int, default=10)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps", type=float, default=1e-6)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
