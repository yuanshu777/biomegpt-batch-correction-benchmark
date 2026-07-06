"""
Prepare real-study-id batch annotation for phase-2 gut samples.

Input:
  BiomeGPT_species_samples_studyIDs.csv

This file maps sample IDs to study_name. It is a better batch/domain label source than
prefix-derived proxies when the goal is study-level batch correction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_study_map(path: Path) -> pd.DataFrame:
    # The uploaded CSV contains an unmatched quote in one study_name; QUOTE_NONE keeps all rows.
    df = pd.read_csv(path, quoting=csv.QUOTE_NONE)
    df = df.rename(columns={df.columns[0]: "sample_id"})
    if "study_name" not in df.columns:
        raise ValueError(f"Expected column 'study_name' in {path}")
    df["sample_id"] = df["sample_id"].astype(str)
    df["real_study_id_raw"] = df["study_name"].astype(str)
    df["real_study_id"] = (
        df["real_study_id_raw"]
        .str.strip()
        .str.strip('"')
        .str.replace(r"\s+", "_", regex=True)
        .str.replace('"', "", regex=False)
    )
    df = df[["sample_id", "real_study_id", "real_study_id_raw"]].drop_duplicates("sample_id")
    return df


def str_bool(x: bool) -> str:
    return "True" if bool(x) else "False"


def run(args: argparse.Namespace) -> None:
    meta = pd.read_csv(args.phase2_meta).rename(columns={"Unnamed: 0": "sample_id"})
    study = load_study_map(Path(args.study_ids_csv))
    out = meta.merge(study, on="sample_id", how="left")

    out["batch_label_external_recommended"] = np.where(
        out["real_study_id"].notna(),
        "real_study:" + out["real_study_id"].astype(str),
        np.nan,
    )
    out["external_confidence"] = np.where(out["real_study_id"].notna(), "high", "low")
    out["external_source"] = np.where(
        out["real_study_id"].notna(),
        "BiomeGPT_species_samples_studyIDs.csv",
        "missing from BiomeGPT_species_samples_studyIDs.csv",
    )
    out["external_study_accession"] = out["real_study_id"]
    out["external_study_title"] = out["real_study_id_raw"]
    out["needs_manual_review"] = out["real_study_id"].isna().map(str_bool)

    pheno_col = "Phenotype_fullname" if "Phenotype_fullname" in out.columns else "Phenotype"
    stats = []
    for label, group in out.dropna(subset=["batch_label_external_recommended"]).groupby("batch_label_external_recommended"):
        counts = group[pheno_col].astype(str).value_counts()
        top_count = int(counts.iloc[0])
        n = int(len(group))
        stats.append(
            {
                "batch_label_external_recommended": label,
                "batch_n_samples": n,
                "batch_n_phenotypes": int(counts.size),
                "batch_top_phenotype": str(counts.index[0]),
                "batch_top_phenotype_count": top_count,
                "batch_top_phenotype_fraction": float(top_count / n),
                "phenotype_confounding_warning": bool((top_count / n) >= args.confounding_threshold or counts.size <= 1),
            }
        )
    stats_df = pd.DataFrame(stats)
    out = out.merge(stats_df, on="batch_label_external_recommended", how="left")
    out["phenotype_confounding_warning"] = out["phenotype_confounding_warning"].fillna(True).map(str_bool)
    out["safe_for_final_batch_correction_conservative"] = (
        out["external_confidence"].eq("high")
        & out["phenotype_confounding_warning"].eq("False")
    ).map(str_bool)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    summary = {
        "input_study_ids_csv": args.study_ids_csv,
        "output_csv": str(output),
        "n_phase2_samples": int(len(out)),
        "n_matched_real_study_id": int(out["real_study_id"].notna().sum()),
        "n_missing_real_study_id": int(out["real_study_id"].isna().sum()),
        "n_unique_real_study_id": int(out["real_study_id"].nunique(dropna=True)),
        "n_batch_labels": int(out["batch_label_external_recommended"].nunique(dropna=True)),
        "external_confidence_counts": out["external_confidence"].value_counts(dropna=False).astype(int).to_dict(),
        "safe_counts": out["safe_for_final_batch_correction_conservative"].value_counts(dropna=False).astype(int).to_dict(),
        "confounding_counts": out["phenotype_confounding_warning"].value_counts(dropna=False).astype(int).to_dict(),
        "min_batch_size": int(args.min_batch_size),
        "n_labels_ge_min_batch_size": int((out["batch_label_external_recommended"].value_counts() >= args.min_batch_size).sum()),
        "n_samples_in_labels_ge_min_batch_size": int(
            out["batch_label_external_recommended"].map(out["batch_label_external_recommended"].value_counts()).ge(args.min_batch_size).sum()
        ),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    stats_df.sort_values(["phenotype_confounding_warning", "batch_top_phenotype_fraction"], ascending=[False, False]).to_csv(
        output.with_suffix(".batch_summary.csv"), index=False
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare real-study-id batch annotation.")
    p.add_argument("--study_ids_csv", default="BiomeGPT_species_samples_studyIDs.csv")
    p.add_argument("--phase2_meta", default="dataset_v3/meta_pretraining_phase2_gut.csv")
    p.add_argument("--output_csv", default="dataset_v3/meta_pretraining_phase2_gut_real_study_annotation.csv")
    p.add_argument("--confounding_threshold", type=float, default=0.80)
    p.add_argument("--min_batch_size", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
