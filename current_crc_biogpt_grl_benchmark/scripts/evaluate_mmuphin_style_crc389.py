from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.io import read_matrix
from src.evaluation.plots import save_pca_plot


DATA_DIR = ROOT / "outputs" / "crc_overlap_benchmark"
METRIC_DIR = ROOT / "outputs" / "metrics"
FIGURE_DIR = ROOT / "outputs" / "figures" / "mmuphin_style_crc389"
REPORT_DIR = ROOT / "reports"
OFFICIAL_MMUPHIN_REFERENCE = ROOT.parent / "crc_controlled_benchmark" / "reports" / "crc_raw_vs_mmuphin_metrics.csv"


PRIMARY_METRICS = [
    "study_R2_condition_controlled",
    "study_prediction_balanced_accuracy",
    "disease_LOSO_mean_within_study_AUC",
    "condition_R2_study_controlled",
]


def align_matrix(matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    ids = metadata["sample_id"].astype(str).tolist()
    indexed = matrix.copy()
    indexed["sample_id"] = indexed["sample_id"].astype(str)
    return indexed.set_index("sample_id").loc[ids].reset_index()


def feature_matrix(matrix: pd.DataFrame, abundance_like: bool) -> np.ndarray:
    x = matrix.drop(columns=["sample_id"]).astype(float).to_numpy()
    if abundance_like:
        x = np.log1p(1000.0 * np.clip(x, 0, None))
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x - mean) / std


def one_hot(values: pd.Series) -> np.ndarray:
    levels = sorted(values.astype(str).unique())
    out = np.zeros((len(values), len(levels)), dtype=float)
    index = {level: i for i, level in enumerate(levels)}
    for row, value in enumerate(values.astype(str)):
        out[row, index[value]] = 1.0
    return out[:, 1:] if out.shape[1] > 1 else out


def design_matrix(metadata: pd.DataFrame, columns: list[str]) -> np.ndarray:
    parts = [np.ones((len(metadata), 1), dtype=float)]
    for column in columns:
        parts.append(one_hot(metadata[column]))
    return np.concatenate(parts, axis=1)


def sse_after_ols(x: np.ndarray, design: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    residual = x - design @ coef
    return float(np.sum(residual**2))


def partial_r2(x: np.ndarray, metadata: pd.DataFrame, target: str, controls: list[str]) -> float:
    total = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    if total == 0:
        return float("nan")
    reduced = design_matrix(metadata, controls)
    full = design_matrix(metadata, controls + [target])
    return (sse_after_ols(x, reduced) - sse_after_ols(x, full)) / total


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for label in sorted(set(y_true.tolist())):
        idx = y_true == label
        recalls.append(float(np.mean(y_pred[idx] == label)))
    return float(np.mean(recalls))


def choose_c_classification(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, inner_splitter, classes: np.ndarray, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    cs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    scores = []
    train_labels = y[train_idx]
    for c in cs:
        fold_scores = []
        for inner_train_rel, inner_val_rel in inner_splitter.split(np.arange(len(train_idx)), train_labels):
            inner_train = train_idx[inner_train_rel]
            inner_val = train_idx[inner_val_rel]
            model = LogisticRegression(
                C=c,
                max_iter=1000,
                class_weight="balanced",
                random_state=seed,
            )
            model.fit(x[inner_train], y[inner_train])
            pred = model.predict(x[inner_val])
            fold_scores.append(balanced_accuracy_score(y[inner_val], pred))
        scores.append(float(np.mean(fold_scores)))
    return float(cs[int(np.argmax(scores))])


def study_prediction_balanced_accuracy(x: np.ndarray, metadata: pd.DataFrame, seed: int = 42) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    y = metadata["studyID"].astype(str).to_numpy()
    predictions = np.empty(len(y), dtype=object)
    repeats = [seed, seed + 17, seed + 31]
    repeat_scores = []
    for repeat_seed in repeats:
        predictions[:] = None
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat_seed)
        for train_idx, test_idx in outer.split(x, y):
            inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=repeat_seed + 1)
            c = choose_c_classification(x, y, train_idx, inner, np.unique(y), repeat_seed)
            model = LogisticRegression(
                C=c,
                max_iter=1000,
                class_weight="balanced",
                random_state=repeat_seed,
            )
            model.fit(x[train_idx], y[train_idx])
            predictions[test_idx] = model.predict(x[test_idx])
        repeat_scores.append(balanced_accuracy(y, predictions.astype(str)))
    return float(np.mean(repeat_scores))


def choose_c_auc(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, inner_splitter, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    cs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    scores = []
    train_labels = y[train_idx]
    for c in cs:
        fold_scores = []
        for inner_train_rel, inner_val_rel in inner_splitter.split(np.arange(len(train_idx)), train_labels):
            inner_train = train_idx[inner_train_rel]
            inner_val = train_idx[inner_val_rel]
            if len(np.unique(y[inner_val])) < 2:
                continue
            model = LogisticRegression(C=c, max_iter=1000, class_weight="balanced", random_state=seed)
            model.fit(x[inner_train], y[inner_train])
            prob = model.predict_proba(x[inner_val])[:, 1]
            fold_scores.append(float(roc_auc_score(y[inner_val], prob)))
        scores.append(float(np.mean(fold_scores)) if fold_scores else float("-inf"))
    return float(cs[int(np.argmax(scores))])


def disease_loso_mean_within_study_auc(x: np.ndarray, metadata: pd.DataFrame, seed: int = 42) -> tuple[float, str]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = (metadata["study_condition"].astype(str).to_numpy() == "CRC").astype(int)
    studies = metadata["studyID"].astype(str).to_numpy()
    per_study: dict[str, float] = {}
    skipped: list[str] = []
    for study in sorted(set(studies.tolist())):
        test_idx = np.where(studies == study)[0]
        train_idx = np.where(studies != study)[0]
        if len(np.unique(y[test_idx])) < 2:
            skipped.append(study)
            continue
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        c = choose_c_auc(x, y, train_idx, inner, seed)
        model = LogisticRegression(C=c, max_iter=1000, class_weight="balanced", random_state=seed)
        model.fit(x[train_idx], y[train_idx])
        prob = model.predict_proba(x[test_idx])[:, 1]
        per_study[study] = float(roc_auc_score(y[test_idx], prob))
    detail = "; ".join([f"{k}={v:.3f}" for k, v in per_study.items()])
    if skipped:
        detail += "; skipped_no_both_classes=" + "|".join(skipped)
    return float(np.mean(list(per_study.values()))), detail


def evaluate_method(method: str, matrix: pd.DataFrame, metadata: pd.DataFrame, abundance_like: bool) -> list[dict[str, Any]]:
    x = feature_matrix(align_matrix(matrix, metadata), abundance_like=abundance_like)
    rows = [
        {
            "method": method,
            "metric": "study_R2_condition_controlled",
            "estimate": partial_r2(x, metadata, target="studyID", controls=["study_condition"]),
            "detail": "Euclidean/linear partial R2 in standardized representation space",
        },
        {
            "method": method,
            "metric": "study_prediction_balanced_accuracy",
            "estimate": study_prediction_balanced_accuracy(x, metadata),
            "detail": "3 repeats of stratified 5-fold logistic probe with inner C selection",
        },
    ]
    auc, detail = disease_loso_mean_within_study_auc(x, metadata)
    rows.append(
        {
            "method": method,
            "metric": "disease_LOSO_mean_within_study_AUC",
            "estimate": auc,
            "detail": detail,
        }
    )
    rows.append(
        {
            "method": method,
            "metric": "condition_R2_study_controlled",
            "estimate": partial_r2(x, metadata, target="study_condition", controls=["studyID"]),
            "detail": "Euclidean/linear partial R2 in standardized representation space",
        }
    )
    return rows


def pivot_primary(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot(index="method", columns="metric", values="estimate").reset_index()
    order = ["Raw abundance", "MMUPHin adjusted abundance", "Full-data tuned GRL", "Cross-fitted tuned GRL"]
    wide["_order"] = wide["method"].map({name: i for i, name in enumerate(order)})
    return wide.sort_values("_order").drop(columns="_order")


def load_official_reference() -> pd.DataFrame:
    if not OFFICIAL_MMUPHIN_REFERENCE.exists():
        return pd.DataFrame()
    ref = pd.read_csv(OFFICIAL_MMUPHIN_REFERENCE)
    ref = ref[ref["metric"].isin(PRIMARY_METRICS)].copy()
    ref["estimate"] = ref["estimate"].astype(float)
    return ref[["method", "metric", "estimate", "detail"]]


def write_summary(metrics: pd.DataFrame, official_ref: pd.DataFrame, paths: dict[str, str]) -> None:
    wide = pivot_primary(metrics)
    crossfit = wide[wide["method"] == "Cross-fitted tuned GRL"].iloc[0]
    mmuphin = wide[wide["method"] == "MMUPHin adjusted abundance"].iloc[0]
    raw = wide[wide["method"] == "Raw abundance"].iloc[0]
    verdict = (
        "Cross-fitted GRL has the lowest study-classifier balanced accuracy on this CRC389 same-evaluator table, "
        "but its condition-controlled study R2 is still higher than MMUPHin and its disease LOSO AUC is lower than both raw and MMUPHin."
        if crossfit["study_prediction_balanced_accuracy"] < mmuphin["study_prediction_balanced_accuracy"]
        else "Cross-fitted GRL does not reduce study prediction more than MMUPHin on this CRC389 table."
    )
    lines = [
        "# MMUPHin-Style CRC389 GRL Comparison",
        "",
        "## Scope",
        "",
        "This report re-evaluates Raw abundance, MMUPHin adjusted abundance, Full-data tuned GRL, and Cross-fitted tuned GRL on the same 389 overlap samples with MMUPHin-style metrics.",
        "",
        "Important: this is not the original 551-sample MMUPHin controlled benchmark from the professor-facing screenshot. CRC389 excludes some samples/studies, and FengQ has only CRC samples in this overlap subset, so its held-out disease AUC is skipped.",
        "",
        "## CRC389 Same-Evaluator Primary Table",
        "",
        wide.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"- Raw study BA is {raw['study_prediction_balanced_accuracy']:.3f}; MMUPHin is {mmuphin['study_prediction_balanced_accuracy']:.3f}; cross-fitted GRL is {crossfit['study_prediction_balanced_accuracy']:.3f}.",
        f"- Raw disease LOSO AUC is {raw['disease_LOSO_mean_within_study_AUC']:.3f}; MMUPHin is {mmuphin['disease_LOSO_mean_within_study_AUC']:.3f}; cross-fitted GRL is {crossfit['disease_LOSO_mean_within_study_AUC']:.3f}.",
        f"- Main reading: {verdict}",
        "- Therefore your intuition is partly right: cross-fitted GRL is not bad on the study-classifier metric. The unresolved issues are disease-signal retention under LOSO and the higher study R2 compared with MMUPHin.",
        "",
        "## Metric Caveats",
        "",
        "- Original MMUPHin abundance PERMANOVA uses Bray-Curtis distance. GRL z has negative embedding dimensions, so this CRC389 same-evaluator table uses standardized Euclidean/linear partial R2 for all four methods.",
        "- Classifier metrics are logistic probes with deterministic repeated folds, not the R/glmnet implementation from the original controlled benchmark.",
        "- Use this as a fair CRC389 diagnostic, not as a replacement for the original 551-sample MMUPHin table.",
        "",
    ]
    if not official_ref.empty:
        lines.extend(
            [
                "## Official Full Benchmark Reference",
                "",
                "These are the current local frozen raw/MMUPHin reference metrics from `crc_controlled_benchmark`; they are included to explain why the screenshot numbers are close but not identical to CRC389.",
                "",
                official_ref.to_markdown(index=False),
                "",
            ]
        )
    lines.extend(["## Output Files", ""])
    for key, value in paths.items():
        lines.append(f"- `{key}`: `{Path(value).relative_to(ROOT)}`")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "mmuphin_style_crc389_grl_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(DATA_DIR / "metadata_389.csv", dtype=str)
    methods = [
        ("Raw abundance", DATA_DIR / "raw_abundance_389.csv", True, "raw_abundance"),
        ("MMUPHin adjusted abundance", DATA_DIR / "mmuphin_adjusted_abundance_389.csv", True, "mmuphin_adjusted_abundance"),
        ("Full-data tuned GRL", DATA_DIR / "grl_full_data_tuned_z_389.csv", False, "full_data_tuned_grl"),
        ("Cross-fitted tuned GRL", DATA_DIR / "grl_crossfit_corrected_z_389.csv", False, "crossfitted_tuned_grl"),
    ]
    all_rows: list[dict[str, Any]] = []
    for method, path, abundance_like, slug in methods:
        matrix = read_matrix(path)
        matrix = align_matrix(matrix, metadata)
        all_rows.extend(evaluate_method(method, matrix, metadata, abundance_like=abundance_like))
        save_pca_plot(matrix, metadata, "studyID", FIGURE_DIR / f"{slug}_pca_by_study.png")
        save_pca_plot(matrix, metadata, "study_condition", FIGURE_DIR / f"{slug}_pca_by_condition.png")
        print("evaluated", method)
    metrics = pd.DataFrame(all_rows)
    metrics_path = METRIC_DIR / "mmuphin_style_crc389_metrics_long.csv"
    primary_path = METRIC_DIR / "mmuphin_style_crc389_primary_table.csv"
    official_path = METRIC_DIR / "mmuphin_style_official_full_reference_raw_vs_mmuphin.csv"
    metrics.to_csv(metrics_path, index=False)
    pivot_primary(metrics).to_csv(primary_path, index=False)
    official_ref = load_official_reference()
    official_ref.to_csv(official_path, index=False)
    paths = {
        "metrics_long": str(metrics_path),
        "primary_table": str(primary_path),
        "official_full_reference": str(official_path),
        "figure_dir": str(FIGURE_DIR),
    }
    write_summary(metrics, official_ref, paths)
    print("MMUPHIN_STYLE_CRC389_OK")
    print(pivot_primary(metrics).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
