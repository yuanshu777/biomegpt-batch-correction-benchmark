from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.plots import save_pca_plot


DATA_DIR = PROJECT_ROOT / "crc_controlled_benchmark" / "data"
METHOD_DIR = PROJECT_ROOT / "crc_controlled_benchmark" / "methods" / "scgpt_biomegpt"
OUTPUT_DIR = ROOT / "outputs" / "condition_amplification_full_crc"
METRIC_DIR = ROOT / "outputs" / "metrics"
FIGURE_DIR = ROOT / "outputs" / "figures" / "condition_amplification_full_crc"
REPORT_DIR = ROOT / "reports"


METHODS = [
    ("raw", DATA_DIR / "crc_raw_abundance.csv"),
    ("mmuphin", DATA_DIR / "crc_mmuphin_adjusted_abundance.csv"),
    ("grl_cw01", METHOD_DIR / "grl_abundance_l8_lam10_cw01_rw1.csv"),
    ("grl_cw001", METHOD_DIR / "grl_abundance_l8_lam10_cw001_rw1.csv"),
    ("grl_cw0", METHOD_DIR / "grl_abundance_l8_lam10_cw0_rw1.csv"),
    ("grl_mech_best", METHOD_DIR / "grl_mech_context_only_l8_lam10_rw5_rel1_var1.csv"),
]


def read_abundance(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    df = pd.read_csv(path)
    features = df["feature"].astype(str).tolist()
    sample_ids = [c for c in df.columns if c != "feature"]
    abundance = df[sample_ids].astype(float).to_numpy()
    return features, sample_ids, abundance


def to_sample_frame(features: list[str], sample_ids: list[str], abundance: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(abundance.T, columns=features)
    out.insert(0, "sample_id", sample_ids)
    return out


def standardized_log_features(abundance: np.ndarray) -> np.ndarray:
    x = np.log1p(1000.0 * np.clip(abundance.T, 0, None)).astype(float)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x - x.mean(axis=0, keepdims=True)) / std


def effective_rank(x: np.ndarray) -> tuple[float, int, float, float]:
    _, s, _ = np.linalg.svd(x - x.mean(axis=0, keepdims=True), full_matrices=False)
    eig = s**2
    eig = eig[eig > 1e-12]
    prop = eig / eig.sum()
    entropy = -float(np.sum(prop * np.log(prop)))
    cumulative = np.cumsum(prop)
    return float(np.exp(entropy)), int(np.searchsorted(cumulative, 0.9) + 1), float(prop[0]), float(prop[:2].sum())


def one_hot(values: pd.Series) -> np.ndarray:
    levels = sorted(values.astype(str).unique())
    out = np.zeros((len(values), len(levels)))
    lookup = {level: i for i, level in enumerate(levels)}
    for i, value in enumerate(values.astype(str)):
        out[i, lookup[value]] = 1.0
    return out[:, 1:] if out.shape[1] > 1 else out


def design(metadata: pd.DataFrame, columns: list[str]) -> np.ndarray:
    parts = [np.ones((len(metadata), 1))]
    for column in columns:
        parts.append(one_hot(metadata[column]))
    return np.concatenate(parts, axis=1)


def partial_r2(x: np.ndarray, metadata: pd.DataFrame, target: str, controls: list[str]) -> float:
    def sse(d: np.ndarray) -> float:
        coef, *_ = np.linalg.lstsq(d, x, rcond=None)
        return float(np.sum((x - d @ coef) ** 2))

    total = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return (sse(design(metadata, controls)) - sse(design(metadata, controls + [target]))) / total


def pc_condition_auc(x: np.ndarray, metadata: pd.DataFrame) -> tuple[float, float]:
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    coords = PCA(n_components=2, random_state=0).fit_transform(x)
    y = (metadata["study_condition"].astype(str).to_numpy() == "CRC").astype(int)
    auc = float(roc_auc_score(y, coords[:, 0]))
    auc = max(auc, 1.0 - auc)
    return auc, float(np.corrcoef(coords[:, 0], y)[0, 1])


def condition_effect_features(features: list[str], x: np.ndarray, metadata: pd.DataFrame, method: str) -> pd.DataFrame:
    condition = metadata["study_condition"].astype(str).to_numpy()
    crc_mean = x[condition == "CRC"].mean(axis=0)
    control_mean = x[condition == "control"].mean(axis=0)
    diff = crc_mean - control_mean
    out = pd.DataFrame({"method": method, "feature": features, "crc_minus_control_zmean": diff})
    out["abs_effect"] = out["crc_minus_control_zmean"].abs()
    return out.sort_values("abs_effect", ascending=False)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(DATA_DIR / "crc_metadata.csv", dtype=str)
    rows: list[dict[str, Any]] = []
    feature_rows: list[pd.DataFrame] = []
    for method, path in METHODS:
        features, sample_ids, abundance = read_abundance(path)
        if sample_ids != metadata["sample_id"].astype(str).tolist():
            raise ValueError(f"Sample order mismatch for {method}")
        x = standardized_log_features(abundance)
        eff_rank, dims90, pc1_var, pc12_var = effective_rank(x)
        pc1_auc, pc1_corr = pc_condition_auc(x, metadata)
        rows.append(
            {
                "method": method,
                "effective_rank": eff_rank,
                "dims_for_90pct_variance": dims90,
                "pc1_variance_fraction": pc1_var,
                "pc1_pc2_variance_fraction": pc12_var,
                "pc1_condition_auc_abs": pc1_auc,
                "pc1_condition_correlation": pc1_corr,
                "condition_R2_study_controlled_linear": partial_r2(x, metadata, "study_condition", ["studyID"]),
                "study_R2_condition_controlled_linear": partial_r2(x, metadata, "studyID", ["study_condition"]),
                "sample_sum_min": float(np.min(abundance.sum(axis=0))),
                "sample_sum_max": float(np.max(abundance.sum(axis=0))),
                "zero_fraction": float(np.mean(abundance <= 0)),
            }
        )
        feature_rows.append(condition_effect_features(features, x, metadata, method).head(25))
        save_pca_plot(to_sample_frame(features, sample_ids, abundance), metadata, "studyID", FIGURE_DIR / f"{method}_pca_by_study.png")
        save_pca_plot(
            to_sample_frame(features, sample_ids, abundance),
            metadata,
            "study_condition",
            FIGURE_DIR / f"{method}_pca_by_condition.png",
        )
        print("diagnosed", method)

    diagnostics = pd.DataFrame(rows)
    top_features = pd.concat(feature_rows, ignore_index=True)
    diagnostics_path = METRIC_DIR / "condition_amplification_full_crc_diagnostics.csv"
    top_features_path = METRIC_DIR / "condition_amplification_full_crc_top_features.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    top_features.to_csv(top_features_path, index=False)

    lines = [
        "# Condition Amplification Diagnostics on Full CRC",
        "",
        "## Scope",
        "",
        "Diagnostics for raw abundance, MMUPHin, and GRL abundance-decoder outputs. The goal is to see whether high disease AUC is accompanied by artificial condition-axis amplification or low-dimensional collapse.",
        "",
        "## Summary",
        "",
        diagnostics.to_markdown(index=False),
        "",
        "## Reading",
        "",
        "- A high `pc1_condition_auc_abs`, very high condition R2, or very small effective rank would suggest condition-coded amplification.",
        "- Compare `grl_cw01` to `grl_cw001` and `grl_cw0`: if stronger condition supervision increases condition R2 sharply, it should be treated cautiously.",
        "",
        "## Output Files",
        "",
        f"- `diagnostics`: `{diagnostics_path.relative_to(ROOT)}`",
        f"- `top_condition_features`: `{top_features_path.relative_to(ROOT)}`",
        f"- `figures`: `{FIGURE_DIR.relative_to(ROOT)}`",
    ]
    (REPORT_DIR / "condition_amplification_full_crc_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("CONDITION_AMPLIFICATION_DIAGNOSTICS_OK")
    print(diagnostics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
