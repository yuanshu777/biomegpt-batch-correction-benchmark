from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.io import read_simple_yaml, resolve_path


def status(path: Path | None) -> tuple[str, str]:
    if path is None:
        return "MISSING", "not configured"
    return ("FOUND", str(path)) if path.exists() else ("MISSING", str(path))


def main() -> int:
    config = read_simple_yaml(ROOT / "configs" / "paths.yaml")
    checks = [
        ("MMUPHin CRC abundance", resolve_path(config.get("crc_raw_abundance"), ROOT)),
        ("MMUPHin CRC metadata", resolve_path(config.get("crc_metadata"), ROOT)),
        ("MMUPHin adjusted abundance", resolve_path(config.get("crc_mmuphin_adjusted_abundance"), ROOT)),
        ("overlap sample file", resolve_path(config.get("crc_sample_overlap"), ROOT)),
        ("overlap study file", resolve_path(config.get("crc_study_overlap"), ROOT)),
        ("overlap counts file", resolve_path(config.get("crc_overlap_counts"), ROOT)),
        ("BiomeGPT checkpoint", resolve_path(config.get("biogpt_checkpoint"), ROOT)),
        ("BiomeGPT checkpoint stage1", resolve_path(config.get("biogpt_checkpoint_stage1"), ROOT)),
        ("BiomeGPT checkpoint stage2", resolve_path(config.get("biogpt_checkpoint_stage2"), ROOT)),
        ("BiomeGPT raw data", resolve_path(config.get("biogpt_raw_abundance"), ROOT)),
        ("BiomeGPT taxonomy xlsx", resolve_path(config.get("biogpt_taxonomy_xlsx"), ROOT)),
        ("BiomeGPT raw CLS embeddings", resolve_path(config.get("biogpt_raw_cls_embeddings"), ROOT)),
        ("GRL module", ROOT / "src" / "grl_correction" / "train_grl.py"),
        ("overlap manifest 389", ROOT / "data_manifest" / "crc_overlap_manifest_389.csv"),
        ("metadata 389", ROOT / "outputs" / "crc_overlap_benchmark" / "metadata_389.csv"),
        ("raw abundance 389", ROOT / "outputs" / "crc_overlap_benchmark" / "raw_abundance_389.csv"),
        ("MMUPHin adjusted abundance 389", ROOT / "outputs" / "crc_overlap_benchmark" / "mmuphin_adjusted_abundance_389.csv"),
    ]
    rows = []
    for name, path in checks:
        found, detail = status(path)
        rows.append({"asset": name, "status": found, "path_or_note": detail})
    for folder in ["outputs/crc_overlap_benchmark", "outputs/figures", "outputs/metrics", "reports"]:
        path = ROOT / folder
        rows.append({"asset": f"required output folder: {folder}", "status": "FOUND" if path.exists() else "MISSING", "path_or_note": str(path)})

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(reports / "asset_check_report.csv", index=False)
    lines = ["# Asset Check Report", "", "| asset | status | path_or_note |", "|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['asset']} | {row['status']} | {row['path_or_note']} |")
    (reports / "asset_check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ASSET_CHECK_COMPLETE")
    print(df["status"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
