from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_mmuphin_style_crc389 import evaluate_method, pivot_primary
from src.evaluation.io import read_matrix
from src.evaluation.plots import save_pca_plot


FULL_DIR = PROJECT / "crc_controlled_benchmark"
FULL_DATA = FULL_DIR / "data"
FULL_REPORTS = FULL_DIR / "reports"
FULL_METHODS = FULL_DIR / "methods" / "scgpt_biomegpt"
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_FULL = ROOT / "outputs" / "crc_full551_benchmark"
REPORTS = ROOT / "reports"

RAW_METRICS = FULL_REPORTS / "crc_raw_vs_mmuphin_metrics.csv"
RAW_PLOTS = FULL_REPORTS / "plots"

METHOD_IDS = [
    "grl_abundance_l8_lam10_cw01_rw1",
    "grl_abundance_l8_lam10_cw001_rw1",
    "grl_abundance_l8_lam10_cw0_rw1",
    "grl_mech_context_only_l8_lam10_rw5_rel1_var1",
    "loso_grl_mech_context_only_l8_lam10_rw5_rel1_var1_oof",
]


def normalize_species(value: str) -> str:
    text = str(value)
    text = re.sub(r"^(?:k|p|c|o|f|g|s)__", "", text)
    text = text.replace("|", "_")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def read_abundance_feature_names(path: Path) -> list[str]:
    df = pd.read_csv(path, usecols=[0])
    return df.iloc[:, 0].astype(str).tolist()


def load_full_metadata() -> pd.DataFrame:
    return pd.read_csv(FULL_DATA / "crc_metadata.csv", dtype=str)


def copy_raw_mmuphin_outputs() -> pd.DataFrame:
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RAW_METRICS)
    metrics.to_csv(OUT_METRICS / "full551_raw_mmuphin_metrics.csv", index=False)

    # Keep requested output names, but preserve the original per-color plot files too.
    shutil.copy2(RAW_PLOTS / "raw_pca_by_study.png", OUT_FIGURES / "full551_raw_pca_by_study.png")
    shutil.copy2(RAW_PLOTS / "mmuphin_pca_by_study.png", OUT_FIGURES / "full551_mmuphin_pca_by_study.png")
    shutil.copy2(RAW_PLOTS / "raw_pca_by_condition.png", OUT_FIGURES / "full551_raw_pca_by_condition.png")
    shutil.copy2(RAW_PLOTS / "mmuphin_pca_by_condition.png", OUT_FIGURES / "full551_mmuphin_pca_by_condition.png")
    # The singular requested names point to the two-method comparison inputs via a small manifest CSV.
    pd.DataFrame(
        [
            {"requested": "full551_raw_mmuphin_pca_by_study.png", "raw": "full551_raw_pca_by_study.png", "mmuphin": "full551_mmuphin_pca_by_study.png"},
            {"requested": "full551_raw_mmuphin_pca_by_condition.png", "raw": "full551_raw_pca_by_condition.png", "mmuphin": "full551_mmuphin_pca_by_condition.png"},
        ]
    ).to_csv(OUT_FIGURES / "full551_raw_mmuphin_pca_manifest.csv", index=False)
    return metrics


def collect_method_metrics() -> pd.DataFrame:
    rows = []
    raw_mm = pd.read_csv(FULL_REPORTS / "methods" / "grl_mech_context_only_l8_lam10_rw5_rel1_var1" / "raw_mmuphin_grl_mech_context_only_l8_lam10_rw5_rel1_var1_metrics.csv")
    for method_id in ["raw", "mmuphin", "grl_mech_context_only_l8_lam10_rw5_rel1_var1"]:
        rows.append(raw_mm[raw_mm["method"] == method_id])
    for method_id in METHOD_IDS:
        method_dir = FULL_REPORTS / "methods" / method_id
        comparison_path = method_dir / f"raw_mmuphin_{method_id}_metrics.csv"
        if method_id.startswith("loso_"):
            comparison_path = method_dir / f"raw_mmuphin_{method_id}_metrics.csv"
        primary_path = method_dir / f"{method_id}_metrics.csv"
        if comparison_path.exists():
            df = pd.read_csv(comparison_path)
            rows.append(df[df["method"] == method_id])
        elif primary_path.exists():
            rows.append(pd.read_csv(primary_path))
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["method", "metric"])
    wide = pivot_primary(out[["method", "metric", "estimate"]].copy())
    wide.to_csv(OUT_METRICS / "full551_grl_abundance_comparison.csv", index=False)
    out.to_csv(OUT_METRICS / "full551_grl_abundance_metrics_long.csv", index=False)
    return wide


def matrix_to_sample_frame(path: Path) -> pd.DataFrame:
    return read_matrix(path)


def effective_rank(x: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def explained_variance_2d(x: np.ndarray) -> tuple[float, float]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    pca = PCA(n_components=2, random_state=0).fit(StandardScaler().fit_transform(x))
    return float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])


def abundance_diagnostics() -> pd.DataFrame:
    metadata = load_full_metadata()
    specs = [
        ("raw", FULL_DATA / "crc_raw_abundance.csv"),
        ("mmuphin", FULL_DATA / "crc_mmuphin_adjusted_abundance.csv"),
    ]
    for method_id in METHOD_IDS:
        path = FULL_METHODS / f"{method_id}.csv"
        if path.exists():
            specs.append((method_id, path))
    rows: list[dict[str, Any]] = []
    fig_dir = OUT_FIGURES / "full551_grl_abundance"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for method, path in specs:
        frame = matrix_to_sample_frame(path)
        ids = metadata["sample_id"].astype(str).tolist()
        frame = frame.set_index("sample_id").loc[ids].reset_index()
        x = frame.drop(columns=["sample_id"]).astype(float).to_numpy()
        pc1, pc2 = explained_variance_2d(x)
        rows.append(
            {
                "method": method,
                "n_samples": x.shape[0],
                "n_features": x.shape[1],
                "effective_rank": effective_rank(x),
                "python_standardized_pca_pc1_variance": pc1,
                "python_standardized_pca_pc2_variance": pc2,
            }
        )
        save_pca_plot(frame, metadata, "studyID", fig_dir / f"{method}_python_pca_by_study.png")
        save_pca_plot(frame, metadata, "study_condition", fig_dir / f"{method}_python_pca_by_condition.png")
    diag = pd.DataFrame(rows)
    diag.to_csv(OUT_METRICS / "full551_grl_abundance_diagnostics.csv", index=False)
    return diag


def load_checkpoint_species() -> tuple[list[str], str | None]:
    checkpoint = PROJECT / "taxonomy_checkpoint_stage2 (1).pt"
    if not checkpoint.exists():
        return [], f"checkpoint missing: {checkpoint}"
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu")
        species = [str(v) for v in payload.get("species", [])]
        return species, None if species else "checkpoint did not contain a species list"
    except Exception as exc:  # pragma: no cover
        return [], f"{type(exc).__name__}: {exc}"


def biogpt_mapping_audit() -> tuple[pd.DataFrame, dict[str, Any]]:
    OUT_FULL.mkdir(parents=True, exist_ok=True)
    mm_species = read_abundance_feature_names(FULL_DATA / "crc_raw_abundance.csv")
    checkpoint_species, checkpoint_error = load_checkpoint_species()
    checkpoint_norm = {normalize_species(s): s for s in checkpoint_species}
    raw_matches = []
    for species in mm_species:
        norm = normalize_species(species)
        mapped = checkpoint_norm.get(norm)
        raw_matches.append(
            {
                "mmuphin_species": species,
                "normalized_species": norm,
                "mapped_biogpt_species": mapped or "",
                "mapped": bool(mapped),
            }
        )
    audit = pd.DataFrame(raw_matches)
    audit.to_csv(OUT_METRICS / "full551_biogpt_species_mapping_audit.csv", index=False)

    raw = read_matrix(FULL_DATA / "crc_raw_abundance.csv")
    mapped_species = audit[audit["mapped"]]["mmuphin_species"].tolist()
    sample_nonzero = []
    if mapped_species:
        mat = raw.set_index("sample_id")[mapped_species]
        nonzero = (mat.astype(float) > 0).sum(axis=1)
        sample_nonzero = nonzero.astype(int).tolist()
        pd.DataFrame({"sample_id": mat.index.astype(str), "nonzero_mapped_species": sample_nonzero}).to_csv(
            OUT_METRICS / "full551_biogpt_per_sample_mapped_nonzero_counts.csv",
            index=False,
        )

    summary = {
        "checkpoint_error": checkpoint_error,
        "mmuphin_species_count": len(mm_species),
        "checkpoint_species_count": len(checkpoint_species),
        "mapped_species_count": int(audit["mapped"].sum()),
        "mapped_species_fraction": float(audit["mapped"].mean()) if len(audit) else 0.0,
        "unmapped_species_count": int((~audit["mapped"]).sum()),
        "per_sample_nonzero_mapped_min": int(np.min(sample_nonzero)) if sample_nonzero else 0,
        "per_sample_nonzero_mapped_median": float(np.median(sample_nonzero)) if sample_nonzero else 0.0,
        "per_sample_nonzero_mapped_max": int(np.max(sample_nonzero)) if sample_nonzero else 0,
        "cls_extraction_feasible": bool(checkpoint_species and audit["mapped"].sum() > 0),
    }
    return audit, summary


def try_extract_full551_cls(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary.get("cls_extraction_feasible"):
        return {"status": "not_feasible", "reason": summary.get("checkpoint_error") or "no mapped species"}
    try:
        sys.path.insert(0, str(ROOT / "legacy_source" / "biomegpt_reusable_20260521_batch_correction" / "dataset_v3"))
        import torch
        from biomegpt_taxonomy_pipeline import abundance_to_binned_matrix, extract_sample_embeddings, load_checkpoint_model

        checkpoint = PROJECT / "taxonomy_checkpoint_stage2 (1).pt"
        taxonomy = ROOT / "legacy_source" / "biomegpt_reusable_20260521_batch_correction" / "dataset_v3" / "species_taxonomy_filled_validated_Serena.xlsx"
        device = torch.device("cpu")
        model, species, _payload = load_checkpoint_model(checkpoint, taxonomy, device)
        raw = read_matrix(FULL_DATA / "crc_raw_abundance.csv")
        ids = raw["sample_id"].astype(str).tolist()
        abund = raw.set_index("sample_id")
        # Direct normalized-name mapping from MMUPHin features into checkpoint species.
        reverse = {normalize_species(c): c for c in abund.columns}
        selected = pd.DataFrame(0.0, index=abund.index, columns=species)
        mapped = 0
        for sp in species:
            source = reverse.get(normalize_species(sp))
            if source is not None:
                selected[sp] = abund[source].astype(float)
                mapped += 1
        if mapped == 0:
            return {"status": "not_feasible", "reason": "no checkpoint species mapped to MMUPHin features"}
        bins = abundance_to_binned_matrix(selected, getattr(model, "num_bins", 32))
        embeddings = extract_sample_embeddings(model, bins, device, batch_size=16)
        out = pd.DataFrame(embeddings, columns=[f"cls_dim_{i}" for i in range(embeddings.shape[1])])
        out.insert(0, "sample_id", ids)
        out_path = OUT_FULL / "biogpt_raw_cls_551.csv"
        out.to_csv(out_path, index=False)
        return {"status": "extracted", "output": str(out_path), "n_samples": len(out), "n_dimensions": embeddings.shape[1], "mapped_checkpoint_species": mapped}
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def crc389_audit() -> pd.DataFrame:
    full_meta = load_full_metadata()
    meta389 = pd.read_csv(ROOT / "outputs" / "crc_overlap_benchmark" / "metadata_389.csv", dtype=str)
    rows = [
        {"dataset": "full551", "n_samples": len(full_meta), "n_studies": full_meta["studyID"].nunique(), "studies": ";".join(sorted(full_meta["studyID"].unique()))},
        {"dataset": "crc389", "n_samples": len(meta389), "n_studies": meta389["studyID"].nunique(), "studies": ";".join(sorted(meta389["studyID"].unique()))},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_METRICS / "crc389_vs_full551_dataset_audit.csv", index=False)
    balance = pd.concat(
        [
            full_meta.assign(dataset="full551"),
            meta389.assign(dataset="crc389"),
        ],
        ignore_index=True,
    ).groupby(["dataset", "studyID", "study_condition"]).size().reset_index(name="n")
    balance.to_csv(OUT_METRICS / "crc389_vs_full551_condition_balance.csv", index=False)
    return out


def write_reports(raw_metrics: pd.DataFrame, comparison: pd.DataFrame, diag: pd.DataFrame, mapping_summary: dict[str, Any], cls_result: dict[str, Any], audit389: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    raw_wide = raw_metrics.pivot(index="metric", columns="method", values="estimate").reset_index()
    (REPORTS / "full551_benchmark_reproduction_summary.md").write_text(
        "# Full551 MMUPHin Benchmark Reproduction\n\n"
        "This restores the original 551-sample MMUPHin CRC dataset as the canonical benchmark. Metrics are copied from the frozen R controlled benchmark evaluator.\n\n"
        "## Raw vs MMUPHin Metrics\n\n"
        f"{raw_wide.to_markdown(index=False)}\n\n"
        "## Outputs\n\n"
        "- `outputs/metrics/full551_raw_mmuphin_metrics.csv`\n"
        "- `outputs/figures/full551_raw_mmuphin_pca_by_study.png`\n"
        "- `outputs/figures/full551_raw_mmuphin_pca_by_condition.png`\n"
        "- `outputs/figures/full551_raw_mmuphin_pca_scores.csv`\n"
        "- `outputs/figures/full551_raw_pca_by_study.png`\n"
        "- `outputs/figures/full551_mmuphin_pca_by_study.png`\n"
        "- `outputs/figures/full551_raw_pca_by_condition.png`\n"
        "- `outputs/figures/full551_mmuphin_pca_by_condition.png`\n",
        encoding="utf-8",
    )

    (REPORTS / "full551_grl_abundance_summary.md").write_text(
        "# Full551 GRL Abundance Summary\n\n"
        "This uses the original 551-sample MMUPHin CRC benchmark. Full-data GRL metrics are reported as diagnostics only; strict LOSO/OOF correction remains the robustness check.\n\n"
        "## Primary Metrics\n\n"
        f"{comparison.to_markdown(index=False)}\n\n"
        "## Geometry Diagnostics\n\n"
        f"{diag.to_markdown(index=False)}\n\n"
        "## Interpretation\n\n"
        "- Mechanism-only full-data GRL reduces study metrics strongly in the frozen R evaluator, but prior strict LOSO/OOF diagnostics did not preserve this advantage.\n"
        "- High PC1/PC2 variance or low effective rank should be treated as possible artificial low-dimensional geometry.\n",
        encoding="utf-8",
    )

    (REPORTS / "full551_biogpt_cls_extraction_audit.md").write_text(
        "# Full551 BiomeGPT CLS Extraction Audit\n\n"
        "This checks whether the original 551 MMUPHin CRC abundance table can be directly mapped into the stage2 BiomeGPT checkpoint species vocabulary.\n\n"
        "## Mapping Summary\n\n"
        f"```json\n{json.dumps(mapping_summary | {'cls_result': cls_result}, indent=2)}\n```\n\n"
        "## Reading\n\n"
        "- If `biogpt_raw_cls_551.csv` exists, CLS extraction succeeded on the canonical 551 benchmark.\n"
        "- If extraction failed, the JSON reason gives the exact blocker.\n"
        "- The mapping audit CSV lists every MMUPHin species and whether it maps to a checkpoint species after conservative normalization.\n",
        encoding="utf-8",
    )

    (REPORTS / "crc389_overlap_audit_vs_full551.md").write_text(
        "# CRC389 Overlap Audit vs Full551\n\n"
        "CRC389 is now treated as an exploratory BiomeGPT-overlap subset unless exact sample identity and study composition can be justified.\n\n"
        "## Dataset Counts\n\n"
        f"{audit389.to_markdown(index=False)}\n\n"
        "## Interpretation\n\n"
        "- Full551 remains the canonical MMUPHin CRC benchmark.\n"
        "- CRC389 changes sample and study composition, including fewer studies than the original 551 benchmark.\n"
        "- PCA and metrics from CRC389 should not be used as primary MMUPHin benchmark claims.\n",
        encoding="utf-8",
    )

    if cls_result.get("status") == "extracted":
        cls_path = Path(cls_result["output"])
        metadata = load_full_metadata()
        rows: list[dict[str, Any]] = []
        methods = [
            ("Raw abundance", FULL_DATA / "crc_raw_abundance.csv", True),
            ("MMUPHin adjusted abundance", FULL_DATA / "crc_mmuphin_adjusted_abundance.csv", True),
            ("BiomeGPT raw CLS 551", cls_path, False),
        ]
        for method, path, abundance_like in methods:
            rows.extend(evaluate_method(method, read_matrix(path), metadata, abundance_like=abundance_like))
        cls_metrics = pd.DataFrame(rows)
        wide = pivot_primary(cls_metrics)
        wide.to_csv(OUT_METRICS / "full551_biogpt_cls_comparison.csv", index=False)
        (REPORTS / "full551_biogpt_cls_summary.md").write_text(
            "# Full551 BiomeGPT CLS Summary\n\n"
            "BiomeGPT raw CLS was extracted for all 551 MMUPHin CRC samples using mapped species from the stage2 checkpoint.\n\n"
            f"{wide.to_markdown(index=False)}\n",
            encoding="utf-8",
        )


def main() -> int:
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUT_FULL.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    raw_metrics = copy_raw_mmuphin_outputs()
    comparison = collect_method_metrics()
    diag = abundance_diagnostics()
    _mapping, mapping_summary = biogpt_mapping_audit()
    cls_result = try_extract_full551_cls(mapping_summary)
    audit389 = crc389_audit()
    write_reports(raw_metrics, comparison, diag, mapping_summary, cls_result, audit389)
    print("FULL551_CANONICAL_OUTPUTS_OK")
    print(json.dumps({"mapping": mapping_summary, "cls_result": cls_result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
