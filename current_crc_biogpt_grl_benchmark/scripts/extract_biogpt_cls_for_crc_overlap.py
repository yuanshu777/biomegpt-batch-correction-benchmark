from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.biogpt_core.checkpoint import MissingAssetError
from src.biogpt_core.extract_cls import extract_cls_from_checkpoint, subset_existing_cls
from src.evaluation.io import read_simple_yaml, resolve_path


def main() -> int:
    config = read_simple_yaml(ROOT / "configs" / "paths.yaml")
    manifest = ROOT / "data_manifest" / "crc_overlap_manifest_389.csv"
    output = ROOT / "outputs" / "crc_overlap_benchmark" / "biogpt_raw_cls_389.csv"
    report = ROOT / "reports" / "biogpt_cls_missing_assets.md"
    raw_cls = resolve_path(config.get("biogpt_raw_cls_embeddings"), ROOT)

    try:
        if raw_cls and raw_cls.exists():
            result = subset_existing_cls(raw_cls, manifest, output)
            print(f"BIOMEGPT_CLS_SUBSET_OK {result}")
            return 0
        result = extract_cls_from_checkpoint(
            checkpoint_path=resolve_path(config.get("biogpt_checkpoint"), ROOT),
            raw_data_path=resolve_path(config.get("biogpt_raw_abundance"), ROOT),
            manifest_path=manifest,
            output_path=output,
            taxonomy_xlsx=resolve_path(config.get("biogpt_taxonomy_xlsx"), ROOT),
            legacy_dataset_v3=ROOT / "legacy_source" / "biomegpt_reusable_20260521_batch_correction" / "dataset_v3",
        )
        print(f"BIOMEGPT_CLS_EXTRACTION_OK {result}")
        return 0
    except Exception as exc:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# BiomeGPT CLS Extraction Asset Report\n\n"
            "CLS extraction was not run to completion.\n\n"
            f"Reason: `{type(exc).__name__}: {exc}`\n\n"
            "Expected output when assets are available:\n\n"
            f"- `{output.relative_to(ROOT)}`\n",
            encoding="utf-8",
        )
        print(f"BIOMEGPT_CLS_EXTRACTION_NOT_RUN: {type(exc).__name__}: {exc}")
        print(f"report: {report}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
