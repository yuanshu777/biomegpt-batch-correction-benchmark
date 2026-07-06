from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.io import read_matrix


def load_crc_metadata(path: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, dtype=str)
    required = {"sample_id", "studyID", "study_condition"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"CRC metadata is missing required columns: {sorted(missing)}")
    return metadata


def load_crc_abundance(path: str | Path) -> pd.DataFrame:
    return read_matrix(path, sample_id_column="sample_id")


def subset_by_samples(matrix: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    present = set(matrix["sample_id"])
    missing = [sample_id for sample_id in sample_ids if sample_id not in present]
    if missing:
        raise ValueError(f"Matrix is missing {len(missing)} requested samples. Example: {missing[:5]}")
    return matrix.set_index("sample_id").loc[sample_ids].reset_index()

