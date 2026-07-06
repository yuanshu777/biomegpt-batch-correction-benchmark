from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.io import read_matrix


def load_overlap_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype=str)
    required = {"sample_id", "biogpt_sample_id", "studyID", "study_condition"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Overlap manifest missing required columns: {sorted(missing)}")
    return manifest


def load_biogpt_abundance(path: str | Path) -> pd.DataFrame:
    return read_matrix(path, sample_id_column="sample_id")


def subset_biogpt_rows(matrix: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    ids = manifest["biogpt_sample_id"].astype(str).tolist()
    matrix = matrix.copy()
    if "sample_id" not in matrix.columns:
        matrix = matrix.rename(columns={matrix.columns[0]: "sample_id"})
    missing = [sample_id for sample_id in ids if sample_id not in set(matrix["sample_id"])]
    if missing:
        raise ValueError(f"BiomeGPT input matrix is missing {len(missing)} overlap samples. Example: {missing[:5]}")
    out = matrix.set_index("sample_id").loc[ids].reset_index()
    out.insert(0, "mmuphin_sample_id", manifest["sample_id"].tolist())
    return out

