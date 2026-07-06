from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.io import read_simple_yaml, resolve_path
from src.mmuphin_bridge.build_overlap import build_overlap_assets


def main() -> int:
    config = read_simple_yaml(ROOT / "configs" / "paths.yaml")
    outputs = build_overlap_assets(
        sample_overlap_csv=resolve_path(config.get("crc_sample_overlap"), ROOT),
        crc_metadata_csv=resolve_path(config.get("crc_metadata"), ROOT),
        raw_abundance_csv=resolve_path(config.get("crc_raw_abundance"), ROOT),
        adjusted_abundance_csv=resolve_path(config.get("crc_mmuphin_adjusted_abundance"), ROOT),
        output_dir=ROOT / "outputs" / "crc_overlap_benchmark",
        manifest_path=ROOT / "data_manifest" / "crc_overlap_manifest_389.csv",
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

