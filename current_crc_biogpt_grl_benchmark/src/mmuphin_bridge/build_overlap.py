from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.io import read_matrix
from .load_crc import load_crc_metadata, subset_by_samples


def build_overlap_manifest(sample_overlap_csv: str | Path, crc_metadata_csv: str | Path) -> pd.DataFrame:
    overlap = pd.read_csv(sample_overlap_csv, dtype=str).fillna("")
    overlap = overlap[overlap["match_status"] != "no_match"].copy()
    metadata = load_crc_metadata(crc_metadata_csv)
    manifest = overlap.merge(
        metadata,
        left_on="mmuphin_sample_id",
        right_on="sample_id",
        how="left",
        suffixes=("", "_crc"),
    )
    if "sample_id" in manifest.columns:
        manifest = manifest.drop(columns=["sample_id"])
    manifest = manifest.rename(
        columns={
            "mmuphin_sample_id": "sample_id",
            "matched_biogpt_sample_ids": "biogpt_sample_id",
            "matched_accession_tokens": "matched_accession_tokens",
        }
    )
    keep = [
        "sample_id",
        "biogpt_sample_id",
        "studyID",
        "study_condition",
        "mmuphin_subjectID",
        "mmuphin_studyID",
        "mmuphin_condition",
        "mmuphin_NCBI_accession",
        "mmuphin_ena_sample_accessions",
        "mmuphin_ena_secondary_sample_accessions",
        "mmuphin_ena_study_accessions",
        "matched_accession_tokens",
        "match_status",
    ]
    keep = [col for col in keep if col in manifest.columns]
    manifest = manifest[keep].drop_duplicates("sample_id").sort_values(["studyID", "sample_id"])
    if len(manifest) != 389:
        raise ValueError(f"Expected 389 overlap samples, found {len(manifest)}.")
    return manifest


def build_overlap_assets(
    sample_overlap_csv: str | Path,
    crc_metadata_csv: str | Path,
    raw_abundance_csv: str | Path | None,
    adjusted_abundance_csv: str | Path | None,
    output_dir: str | Path,
    manifest_path: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_overlap_manifest(sample_overlap_csv, crc_metadata_csv)
    manifest.to_csv(manifest_path, index=False)
    metadata = manifest[["sample_id", "studyID", "study_condition"]].copy()
    metadata.to_csv(output_dir / "metadata_389.csv", index=False)

    sample_ids = manifest["sample_id"].tolist()
    outputs = {
        "manifest": str(manifest_path),
        "metadata": str(output_dir / "metadata_389.csv"),
    }
    if raw_abundance_csv and Path(raw_abundance_csv).exists():
        raw = subset_by_samples(read_matrix(raw_abundance_csv), sample_ids)
        raw.to_csv(output_dir / "raw_abundance_389.csv", index=False)
        outputs["raw_abundance"] = str(output_dir / "raw_abundance_389.csv")
    if adjusted_abundance_csv and Path(adjusted_abundance_csv).exists():
        adjusted = subset_by_samples(read_matrix(adjusted_abundance_csv), sample_ids)
        adjusted.to_csv(output_dir / "mmuphin_adjusted_abundance_389.csv", index=False)
        outputs["mmuphin_adjusted_abundance"] = str(output_dir / "mmuphin_adjusted_abundance_389.csv")
    return outputs
