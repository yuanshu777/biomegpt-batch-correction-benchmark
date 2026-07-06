from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from .checkpoint import MissingAssetError, require_asset
from .data import load_biogpt_abundance, load_overlap_manifest


def subset_existing_cls(raw_cls_path: str | Path, manifest_path: str | Path, output_path: str | Path) -> dict[str, str | int]:
    """Subset an existing sample x dimension CLS matrix to the CRC overlap."""
    raw_cls = pd.read_csv(raw_cls_path)
    if "sample_id" not in raw_cls.columns:
        raw_cls = raw_cls.rename(columns={raw_cls.columns[0]: "sample_id"})
    manifest = load_overlap_manifest(manifest_path)
    mapping = manifest[["sample_id", "biogpt_sample_id"]].copy()
    joined = mapping.merge(raw_cls, left_on="biogpt_sample_id", right_on="sample_id", how="left", suffixes=("_mmuphin", "_biogpt"))
    missing = joined[joined.filter(regex=r"^(?!sample_id|biogpt_sample_id)").isna().all(axis=1)]
    if not missing.empty:
        raise MissingAssetError(f"Existing CLS file is missing {len(missing)} overlap samples.")
    feature_cols = [c for c in raw_cls.columns if c != "sample_id"]
    out = joined[["sample_id_mmuphin"] + feature_cols].rename(columns={"sample_id_mmuphin": "sample_id"})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return {"output": str(output_path), "n_samples": len(out), "n_dimensions": len(feature_cols)}


def extract_cls_from_checkpoint(
    *,
    checkpoint_path: str | Path | None,
    raw_data_path: str | Path | None,
    manifest_path: str | Path,
    output_path: str | Path,
    taxonomy_xlsx: str | Path | None,
    legacy_dataset_v3: str | Path,
    device: str = "cpu",
    batch_size: int = 16,
) -> dict[str, str | int]:
    """Extract sample-prompt/CLS embeddings using the legacy taxonomy model code.

    This uses the old package only for the model definition and checkpoint
    loader. It still emits the focused benchmark schema with MMUPHin sample IDs.
    """
    checkpoint = require_asset(checkpoint_path, "BiomeGPT checkpoint")
    raw_data = require_asset(raw_data_path, "BiomeGPT raw abundance/data")
    taxonomy = require_asset(taxonomy_xlsx, "BiomeGPT taxonomy xlsx")
    manifest_file = require_asset(manifest_path, "CRC overlap manifest")
    legacy_path = require_asset(legacy_dataset_v3, "legacy dataset_v3 source directory")

    sys.path.insert(0, str(legacy_path))
    try:
        import torch
        from biomegpt_taxonomy_pipeline import (
            abundance_to_binned_matrix,
            extract_sample_embeddings,
            load_checkpoint_model,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch and legacy BiomeGPT taxonomy code are required for checkpoint-based CLS extraction.") from exc

    manifest = load_overlap_manifest(manifest_file)
    abund = load_biogpt_abundance(raw_data)
    if "sample_id" not in abund.columns:
        raise ValueError("BiomeGPT abundance matrix must include sample_id after loading.")
    abund = abund.set_index("sample_id")
    wanted = manifest["biogpt_sample_id"].astype(str).tolist()
    missing = [sample_id for sample_id in wanted if sample_id not in abund.index]
    if missing:
        raise MissingAssetError(f"BiomeGPT raw abundance is missing {len(missing)} overlap samples. Example: {missing[:5]}")

    torch_device = torch.device(device)
    model, species, _payload = load_checkpoint_model(checkpoint, taxonomy, torch_device)
    selected = abund.loc[wanted].reindex(columns=species, fill_value=0.0)
    bins = abundance_to_binned_matrix(selected, getattr(model, "num_bins", 32))
    embeddings = extract_sample_embeddings(model, bins, torch_device, batch_size)

    out = pd.DataFrame(embeddings, columns=[f"cls_dim_{i}" for i in range(embeddings.shape[1])])
    out.insert(0, "sample_id", manifest["sample_id"].tolist())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return {"output": str(output_path), "n_samples": len(out), "n_dimensions": embeddings.shape[1]}
