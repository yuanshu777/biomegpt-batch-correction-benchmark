from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.io import read_matrix


DATA_DIR = ROOT / "outputs" / "crc_overlap_benchmark"
FIGURE_DIR = ROOT / "outputs" / "figures" / "mechanism_grl_crc389"


def align_matrix(matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    ids = metadata["sample_id"].astype(str).tolist()
    out = matrix.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    return out.set_index("sample_id").loc[ids].reset_index()


def pca_scores(matrix: pd.DataFrame, metadata: pd.DataFrame, color_column: str, method: str) -> tuple[pd.DataFrame, tuple[float, float]]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for PCA plots.") from exc

    joined = metadata[["sample_id", color_column]].merge(matrix, on="sample_id", how="inner")
    x = joined.drop(columns=["sample_id", color_column]).astype(float).to_numpy()
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(StandardScaler().fit_transform(x))
    scores = pd.DataFrame(
        {
            "sample_id": joined["sample_id"].astype(str).to_numpy(),
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            color_column: joined[color_column].astype(str).to_numpy(),
            "method": method,
        }
    )
    variance = (float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1]))
    return scores, variance


def save_panel(raw: pd.DataFrame, mech: pd.DataFrame, metadata: pd.DataFrame, color_column: str, output_path: Path) -> pd.DataFrame:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for PCA plots.") from exc

    raw_scores, raw_var = pca_scores(raw, metadata, color_column, "Raw abundance")
    mech_scores, mech_var = pca_scores(mech, metadata, color_column, "Mechanism-only GRL abundance")
    levels = sorted(pd.concat([raw_scores[color_column], mech_scores[color_column]]).unique())
    palette = {
        "CRC": "#D55E00",
        "control": "#0072B2",
    }
    fallback = plt.get_cmap("tab10")
    colors = {level: palette.get(level, fallback(i % 10)) for i, level in enumerate(levels)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=160)
    panels = [
        (axes[0], raw_scores, raw_var, "Raw abundance"),
        (axes[1], mech_scores, mech_var, "Mechanism-only GRL abundance"),
    ]
    for ax, scores, var, title in panels:
        for level in levels:
            group = scores[scores[color_column] == level]
            if group.empty:
                continue
            ax.scatter(group["PC1"], group["PC2"], s=18, alpha=0.78, label=level, color=colors[level])
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({100 * var[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100 * var[1]:.1f}%)")
    axes[1].legend(title=color_column, fontsize=7, title_fontsize=8, frameon=False, loc="best")
    fig.suptitle(f"CRC389 Python PCA colored by {color_column}", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return pd.concat([raw_scores, mech_scores], ignore_index=True)


def main() -> int:
    metadata = pd.read_csv(DATA_DIR / "metadata_389.csv", dtype=str)
    raw = align_matrix(read_matrix(DATA_DIR / "raw_abundance_389.csv"), metadata)
    mech = align_matrix(read_matrix(DATA_DIR / "mechanism_grl_abundance_389.csv"), metadata)
    study_scores = save_panel(
        raw,
        mech,
        metadata,
        "studyID",
        FIGURE_DIR / "raw_vs_mechanism_grl_abundance_python_pca_by_study_panel.png",
    )
    condition_scores = save_panel(
        raw,
        mech,
        metadata,
        "study_condition",
        FIGURE_DIR / "raw_vs_mechanism_grl_abundance_python_pca_by_condition_panel.png",
    )
    study_scores.to_csv(FIGURE_DIR / "raw_vs_mechanism_grl_python_pca_by_study_scores.csv", index=False)
    condition_scores.to_csv(FIGURE_DIR / "raw_vs_mechanism_grl_python_pca_by_condition_scores.csv", index=False)
    print("CRC389_PYTHON_PCA_RAW_VS_MECHANISM_OK")
    print(FIGURE_DIR / "raw_vs_mechanism_grl_abundance_python_pca_by_study_panel.png")
    print(FIGURE_DIR / "raw_vs_mechanism_grl_abundance_python_pca_by_condition_panel.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
