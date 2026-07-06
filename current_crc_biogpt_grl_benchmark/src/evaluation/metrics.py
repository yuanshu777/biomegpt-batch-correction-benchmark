from __future__ import annotations

import numpy as np


def balanced_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    for label in sorted(set(y_true.tolist())):
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)) if mask.any() else 0.0)
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for label in sorted(set(y_true.tolist()) | set(y_pred.tolist())):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def binary_auroc(y_true, scores, positive_label="CRC") -> float:
    y = np.asarray([1 if v == positive_label else 0 for v in y_true])
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for value in pos:
        wins += np.sum(value > neg) + 0.5 * np.sum(value == neg)
    return float(wins / (len(pos) * len(neg)))


def describe_vector(values) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

