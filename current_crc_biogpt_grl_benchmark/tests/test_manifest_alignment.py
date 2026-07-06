from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_overlap_manifest_alignment_if_built():
    manifest = ROOT / "data_manifest" / "crc_overlap_manifest_389.csv"
    metadata = ROOT / "outputs" / "crc_overlap_benchmark" / "metadata_389.csv"
    if not manifest.exists() or not metadata.exists():
        return
    manifest_df = pd.read_csv(manifest, dtype=str)
    metadata_df = pd.read_csv(metadata, dtype=str)
    assert len(manifest_df) == 389
    assert len(metadata_df) == 389
    assert manifest_df["sample_id"].is_unique
    assert metadata_df["sample_id"].is_unique
    assert set(manifest_df["sample_id"]) == set(metadata_df["sample_id"])
    assert {"sample_id", "studyID", "study_condition"}.issubset(metadata_df.columns)

