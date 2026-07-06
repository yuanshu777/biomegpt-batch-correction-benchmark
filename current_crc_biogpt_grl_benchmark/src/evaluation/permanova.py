from __future__ import annotations

import numpy as np
import pandas as pd


def distance_r2(matrix: pd.DataFrame, metadata: pd.DataFrame, label_column: str, sample_id_column: str = "sample_id") -> float:
    """Lightweight pseudo-R2 based on group centroids in Euclidean space.

    This is not a replacement for formal PERMANOVA. It is a local fallback used
    when R/vegan is unavailable; formal PERMANOVA should be run for final claims.
    """
    joined = metadata[[sample_id_column, label_column]].merge(matrix, on=sample_id_column, how="inner")
    x = joined.drop(columns=[sample_id_column, label_column]).astype(float).to_numpy()
    y = joined[label_column].astype(str).to_numpy()
    grand = x.mean(axis=0, keepdims=True)
    total = float(np.sum((x - grand) ** 2))
    if total == 0:
        return float("nan")
    between = 0.0
    for label in sorted(set(y.tolist())):
        group = x[y == label]
        between += len(group) * float(np.sum((group.mean(axis=0, keepdims=True) - grand) ** 2))
    return between / total

