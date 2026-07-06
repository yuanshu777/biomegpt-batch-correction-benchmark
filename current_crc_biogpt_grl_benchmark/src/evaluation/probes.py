from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import balanced_accuracy, binary_auroc, macro_f1


def _require_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("scikit-learn is required for probe evaluation.") from exc
    return LogisticRegression, StratifiedKFold, cross_val_predict, make_pipeline, StandardScaler


def probe_classification(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    label_column: str,
    sample_id_column: str = "sample_id",
    positive_label: str | None = None,
    n_splits: int = 5,
    random_state: int = 7,
) -> dict[str, float | str]:
    LogisticRegression, StratifiedKFold, cross_val_predict, make_pipeline, StandardScaler = _require_sklearn()
    joined = metadata[[sample_id_column, label_column]].merge(matrix, on=sample_id_column, how="inner")
    if joined.empty:
        raise ValueError(f"No aligned samples for label {label_column}.")
    y = joined[label_column].astype(str).to_numpy()
    x = joined.drop(columns=[sample_id_column, label_column]).astype(float).to_numpy()
    class_counts = pd.Series(y).value_counts()
    splits = max(2, min(n_splits, int(class_counts.min())))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    pred = cross_val_predict(model, x, y, cv=cv, method="predict")
    out: dict[str, float | str] = {
        "label_column": label_column,
        "n_samples": int(len(y)),
        "n_classes": int(len(class_counts)),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "macro_f1": macro_f1(y, pred),
    }
    if positive_label is not None and len(class_counts) == 2:
        probs = cross_val_predict(model, x, y, cv=cv, method="predict_proba")
        fitted_labels = np.unique(y)
        pos_index = list(fitted_labels).index(positive_label) if positive_label in fitted_labels else 1
        out["auroc"] = binary_auroc(y, probs[:, pos_index], positive_label=positive_label)
    return out
