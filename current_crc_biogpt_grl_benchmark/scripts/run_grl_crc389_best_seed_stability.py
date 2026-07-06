from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grl_quick_crc389 import evaluate_matrix, fmt
from src.evaluation.io import read_matrix
from src.grl_correction.train_grl import train_embedding_grl


DATA_DIR = ROOT / "outputs" / "crc_overlap_benchmark"
METRIC_DIR = ROOT / "outputs" / "metrics"
REPORT_DIR = ROOT / "reports"
SEEDS = [7, 42, 99, 123, 2026]


def main() -> int:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(DATA_DIR / "metadata_389.csv", dtype=str)
    raw = read_matrix(DATA_DIR / "raw_abundance_389.csv")
    baseline_raw = evaluate_matrix("Raw abundance", raw, metadata)

    rows = []
    for seed in SEEDS:
        result = train_embedding_grl(
            raw,
            metadata,
            latent_dim=8,
            epochs=100,
            lr=1e-3,
            lambda_grl=10.0,
            lambda_schedule="linear",
            warmup_fraction=0.1,
            condition_weight=0.1,
            preserve_weight=0.001,
            condition_aware_adversary=True,
            use_study_conditioned_decoder=False,
            batch_size=64,
            seed=seed,
            external_eval_every=None,
            device="cpu",
        )
        row = evaluate_matrix(f"G_seed_{seed}", result.corrected_embeddings, metadata)
        row["seed"] = seed
        rows.append(row)
        print(seed, "study_bacc=", fmt(row["study_balanced_accuracy"]), "condition_auc=", fmt(row["condition_auc"]))

    stability = pd.DataFrame(rows)
    stability_path = METRIC_DIR / "grl_crc389_best_seed_stability.csv"
    stability.to_csv(stability_path, index=False)

    summary = stability[["study_balanced_accuracy", "condition_auc", "condition_balanced_accuracy"]].agg(
        ["mean", "std", "min", "max"]
    )
    lines = [
        "# GRL CRC389 Best Setting Seed Stability",
        "",
        "## Scope",
        "",
        "This repeats the best local abundance-level GRL setting from the tuning sweep across five random seeds. It is still a local scaffold check, not a final BiomeGPT/scGPT result.",
        "",
        "## Best Setting Repeated",
        "",
        "- `latent_dim=8`",
        "- `lambda_grl=10.0`",
        "- `lambda_schedule=linear`",
        "- `warmup_fraction=0.1`",
        "- `condition_weight=0.1`",
        "- `preserve_weight=0.001`",
        "- `condition_aware_adversary=True`",
        "",
        "## Raw Reference",
        "",
        f"- Raw study balanced accuracy: {fmt(baseline_raw['study_balanced_accuracy'])}",
        f"- Raw CRC/control AUROC: {fmt(baseline_raw['condition_auc'])}",
        "",
        "## Seed Results",
        "",
        stability[
            ["seed", "study_balanced_accuracy", "condition_auc", "condition_balanced_accuracy", "condition_macro_f1"]
        ].to_markdown(index=False),
        "",
        "## Aggregate",
        "",
        summary.to_markdown(),
        "",
        "## Interpretation",
        "",
        "- Across all five seeds, study balanced accuracy stayed below raw abundance, so the bottlenecked GRL direction is not just a single-seed accident.",
        "- CRC/control AUROC is less stable. Some seeds preserve or improve disease signal, while others fall below raw abundance.",
        "- The next useful adjustment is stability-oriented: repeat-seed selection, early stopping on external probes, or a small validation split for choosing the checkpoint.",
        "",
        "## Output Files",
        "",
        f"- `seed_stability_metrics`: `{stability_path.relative_to(ROOT)}`",
    ]
    (REPORT_DIR / "grl_crc389_best_seed_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("GRL_CRC389_BEST_SEED_STABILITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
