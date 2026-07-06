"""
Batch-effect diagnostics for taxonomy-aware BiomeGPT sample embeddings.

This script is intentionally diagnostic first:
- extract phase-2 gut sample <cls> embeddings from a pretrained checkpoint
- train shallow probes for batch, phenotype, and Healthy-vs-Diseased labels
- quantify batch/phenotype confounding before any adversarial correction

The outputs are meant to answer whether batch correction is scientifically safe to
attempt, not to claim correction has already been achieved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch

from biomegpt_taxonomy_pipeline import (
    abundance_to_binned_matrix,
    extract_sample_embeddings,
    load_checkpoint_model,
    load_csv_or_zip,
)


def str_to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def make_hd_labels(phenotype: pd.Series) -> pd.Series:
    return phenotype.astype(str).str.lower().ne("healthy").map({False: "Healthy", True: "Diseased"})


def keep_labels_with_min_count(labels: pd.Series, min_count: int) -> pd.Series:
    counts = labels.value_counts(dropna=False)
    return labels.where(labels.map(counts).ge(min_count))


def multiclass_probe(
    x: np.ndarray,
    labels: pd.Series,
    label_name: str,
    seed: int,
    test_size: float,
    min_classes: int = 2,
) -> Dict[str, object]:
    labels = labels.astype(str)
    valid = labels.notna() & labels.ne("nan") & labels.ne("None")
    x = x[valid.to_numpy()]
    y_raw = labels[valid].to_numpy()
    counts = pd.Series(y_raw).value_counts()
    if len(counts) < min_classes:
        return {
            "label_name": label_name,
            "status": "skipped",
            "reason": f"fewer than {min_classes} classes after filtering",
            "n_samples": int(len(y_raw)),
            "n_classes": int(len(counts)),
        }

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    stratify = y if counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
            random_state=seed,
        ),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    dummy_pred = dummy.predict(x_test)

    return {
        "label_name": label_name,
        "status": "ok",
        "n_samples": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(len(encoder.classes_)),
        "min_class_count": int(counts.min()),
        "max_class_count": int(counts.max()),
        "top_classes": counts.head(15).astype(int).to_dict(),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "dummy_accuracy": float(accuracy_score(y_test, dummy_pred)),
        "dummy_balanced_accuracy": float(balanced_accuracy_score(y_test, dummy_pred)),
        "dummy_macro_f1": float(f1_score(y_test, dummy_pred, average="macro")),
    }


def cramers_v(table: pd.DataFrame) -> float:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2, _p, _dof, _expected = chi2_contingency(table.to_numpy())
    n = table.to_numpy().sum()
    denom = n * max(1, min(table.shape[0] - 1, table.shape[1] - 1))
    return float(np.sqrt(chi2 / denom))


def confounding_summary(
    df: pd.DataFrame,
    batch_col: str,
    phenotype_col: str,
    min_batch_size: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    work = df[[batch_col, phenotype_col]].dropna().copy()
    work[batch_col] = keep_labels_with_min_count(work[batch_col].astype(str), min_batch_size)
    work = work.dropna()
    table = pd.crosstab(work[batch_col], work[phenotype_col].astype(str))
    top_rows = []
    for batch, row in table.iterrows():
        n = int(row.sum())
        top_pheno = str(row.idxmax())
        top_count = int(row.max())
        top_rows.append(
            {
                "batch_label": batch,
                "n_samples": n,
                "n_phenotypes": int((row > 0).sum()),
                "top_phenotype": top_pheno,
                "top_phenotype_count": top_count,
                "top_phenotype_fraction": float(top_count / max(n, 1)),
            }
        )
    by_batch = pd.DataFrame(top_rows).sort_values(
        ["top_phenotype_fraction", "n_samples"], ascending=[False, False]
    )
    fractions = by_batch["top_phenotype_fraction"] if len(by_batch) else pd.Series(dtype=float)
    summary = {
        "n_samples": int(len(work)),
        "n_batches": int(table.shape[0]),
        "n_phenotypes": int(table.shape[1]),
        "cramers_v": cramers_v(table),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(work[batch_col].astype(str), work[phenotype_col].astype(str))
        ),
        "top_phenotype_fraction_mean": float(fractions.mean()) if len(fractions) else float("nan"),
        "top_phenotype_fraction_median": float(fractions.median()) if len(fractions) else float("nan"),
        "batches_top_fraction_ge_0_8": int((fractions >= 0.8).sum()) if len(fractions) else 0,
        "batches_top_fraction_eq_1": int((fractions >= 1.0).sum()) if len(fractions) else 0,
    }
    return summary, by_batch


def save_pca_plots(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    out_dir: Path,
    color_columns: Iterable[str],
    max_legend: int = 16,
) -> None:
    import matplotlib.pyplot as plt

    coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    pd.DataFrame(
        {
            "sample_id": meta.index.to_numpy(),
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
        }
    ).to_csv(out_dir / "sample_prompt_pca_coordinates.csv", index=False)

    for col in color_columns:
        if col not in meta.columns:
            continue
        labels = meta[col].astype(str)
        top = labels.value_counts().head(max_legend).index
        plot_labels = labels.where(labels.isin(top), other="Other")
        groups = sorted(plot_labels.unique())
        cmap = plt.get_cmap("tab20")
        plt.figure(figsize=(8, 6))
        for i, group in enumerate(groups):
            idx = plot_labels.eq(group).to_numpy()
            plt.scatter(coords[idx, 0], coords[idx, 1], s=8, alpha=0.75, color=cmap(i % 20), label=group)
        plt.title(f"Phase2 sample prompts PCA colored by {col}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(markerscale=2, fontsize=7, frameon=False, loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / f"sample_prompt_pca_by_{col}.png", dpi=220)
        plt.close()


def load_or_extract_embeddings(args: argparse.Namespace, out_dir: Path) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    cache_npz = out_dir / "phase2_sample_prompt_embeddings.npz"
    cache_meta = out_dir / "phase2_sample_prompt_metadata.csv"
    if cache_npz.exists() and cache_meta.exists() and not args.force_recompute_embeddings:
        payload = np.load(cache_npz, allow_pickle=True)
        emb = payload["embeddings"]
        species = payload["species"].tolist()
        meta = pd.read_csv(cache_meta, index_col=0)
        return emb, meta, species

    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    model, species, _payload = load_checkpoint_model(
        Path(args.checkpoint),
        Path(args.taxonomy_xlsx),
        device,
    )

    abund = load_csv_or_zip(data_dir / "abund_pretraining_phase2_gut.csv.zip")
    batch_meta = pd.read_csv(args.batch_annotation_csv, low_memory=False).set_index("sample_id")
    abund = abund.loc[abund.index.intersection(batch_meta.index)]
    batch_meta = batch_meta.loc[abund.index].copy()
    abund = abund.reindex(columns=species, fill_value=0.0)
    bins = abundance_to_binned_matrix(abund, args.bins)

    emb = extract_sample_embeddings(model, bins, device, args.embedding_batch_size)
    np.savez_compressed(cache_npz, embeddings=emb, sample_ids=abund.index.to_numpy(), species=np.array(species))
    batch_meta.to_csv(cache_meta)
    return emb, batch_meta, species


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings, meta, species = load_or_extract_embeddings(args, out_dir)
    phenotype_col = "Phenotype_fullname" if "Phenotype_fullname" in meta.columns else "Phenotype"
    meta["hd_label"] = make_hd_labels(meta[phenotype_col])

    panels = {
        "batch_high_medium_min_size": meta["external_confidence"].isin(["high", "medium"]),
        "batch_high_only_min_size": meta["external_confidence"].eq("high"),
        "batch_safe_conservative_min_size": str_to_bool_series(meta["safe_for_final_batch_correction_conservative"]),
    }

    probes: List[Dict[str, object]] = []
    confounding: Dict[str, object] = {}
    for panel_name, panel_mask in panels.items():
        panel_meta = meta.loc[panel_mask].copy()
        panel_x = embeddings[panel_mask.to_numpy()]
        batch_labels = keep_labels_with_min_count(
            panel_meta["batch_label_external_recommended"].astype(str),
            args.min_batch_size,
        )
        probes.append(
            multiclass_probe(
                panel_x,
                batch_labels,
                label_name=panel_name,
                seed=args.seed,
                test_size=args.test_size,
            )
        )
        merged = panel_meta.assign(batch_label_for_panel=batch_labels).dropna(subset=["batch_label_for_panel"])
        summary, by_batch = confounding_summary(
            merged,
            "batch_label_for_panel",
            phenotype_col,
            args.min_batch_size,
        )
        confounding[panel_name] = summary
        by_batch.to_csv(out_dir / f"{panel_name}_phenotype_confounding_by_batch.csv", index=False)

    probes.append(
        multiclass_probe(
            embeddings,
            meta[phenotype_col].astype(str),
            label_name="phenotype_fullname",
            seed=args.seed,
            test_size=args.test_size,
        )
    )
    probes.append(
        multiclass_probe(
            embeddings,
            meta["hd_label"],
            label_name="healthy_vs_diseased",
            seed=args.seed,
            test_size=args.test_size,
        )
    )

    safe_mask = str_to_bool_series(meta["safe_for_final_batch_correction_conservative"])
    safe_summary = {
        "n_phase2_samples": int(len(meta)),
        "embedding_shape": list(embeddings.shape),
        "checkpoint": str(args.checkpoint),
        "n_checkpoint_species": int(len(species)),
        "batch_label_unique_total": int(meta["batch_label_external_recommended"].nunique(dropna=True)),
        "external_confidence_counts": meta["external_confidence"].value_counts(dropna=False).astype(int).to_dict(),
        "safe_for_final_batch_correction_conservative_counts": safe_mask.value_counts(dropna=False).astype(int).to_dict(),
        "phenotype_counts": meta[phenotype_col].value_counts(dropna=False).astype(int).to_dict(),
        "hd_counts": meta["hd_label"].value_counts(dropna=False).astype(int).to_dict(),
    }

    out = {
        "run_config": vars(args),
        "data_summary": safe_summary,
        "probe_results": probes,
        "batch_phenotype_confounding": confounding,
        "interpretation": {
            "batch_probe_rule": "High balanced accuracy/macro-F1 means sample prompts encode batch labels.",
            "correction_safety_rule": (
                "Do not adversarially remove batch labels that are strongly phenotype-confounded; "
                "prefer high-confidence and conservative-safe labels for final correction."
            ),
        },
    }
    with open(out_dir / "batch_effect_diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(probes).to_csv(out_dir / "probe_metrics.csv", index=False)
    meta.to_csv(out_dir / "phase2_sample_prompt_metadata_with_batch.csv")
    save_pca_plots(
        embeddings,
        meta,
        out_dir,
        ["external_confidence", "batch_label_external_recommended", phenotype_col, "hd_label"],
    )

    print(json.dumps(out, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch diagnostics from taxonomy-aware BiomeGPT sample prompts")
    p.add_argument("--data_dir", default="dataset_v3")
    p.add_argument("--taxonomy_xlsx", default="dataset_v3/species_taxonomy_filled_validated_Serena.xlsx")
    p.add_argument("--checkpoint", default="dataset_v3/outputs_taxonomy_notebook/taxonomy_checkpoint_stage2.pt")
    p.add_argument(
        "--batch_annotation_csv",
        default="dataset_v3/meta_pretraining_phase2_gut_batch_annotation_external_enriched.csv",
    )
    p.add_argument("--output_dir", default="dataset_v3/outputs_batch_diagnostics_taxonomy")
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embedding_batch_size", type=int, default=16)
    p.add_argument("--min_batch_size", type=int, default=10)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_recompute_embeddings", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
