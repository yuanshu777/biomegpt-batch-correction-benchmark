from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_pca_plot(matrix: pd.DataFrame, metadata: pd.DataFrame, color_column: str, output_path: str | Path, sample_id_column: str = "sample_id") -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib and scikit-learn are required for PCA plots.") from exc

    joined = metadata[[sample_id_column, color_column]].merge(matrix, on=sample_id_column, how="inner")
    x = joined.drop(columns=[sample_id_column, color_column]).astype(float).to_numpy()
    coords = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(x))
    plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], color_column: joined[color_column].astype(str)})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    for label, group in plot_df.groupby(color_column):
        ax.scatter(group["PC1"], group["PC2"], s=18, alpha=0.8, label=label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"PCA colored by {color_column}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

