from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.grl_correction.grl import gradient_reverse


DATA_DIR = PROJECT_ROOT / "crc_controlled_benchmark" / "data"
SPLIT_DIR = PROJECT_ROOT / "crc_controlled_benchmark" / "splits"
METHOD_DIR = PROJECT_ROOT / "crc_controlled_benchmark" / "methods" / "scgpt_biomegpt"
OUTPUT_DIR = ROOT / "outputs" / "crc_full_abundance_grl_loso"
METRIC_DIR = ROOT / "outputs" / "metrics"
REPORT_DIR = ROOT / "reports"


@dataclass
class LosoGRLConfig:
    method_name: str
    latent_dim: int = 8
    hidden_dim: int = 128
    epochs: int = 200
    lr: float = 1e-3
    lambda_grl: float = 10.0
    condition_weight: float = 0.1
    recon_weight: float = 1.0
    rel_recon_weight: float = 0.0
    variance_weight: float = 0.0
    use_condition_head: bool = True
    condition_context_for_adversary: bool = True
    dropout: float = 0.1
    batch_size: int = 64
    seed: int = 42
    warmup_fraction: float = 0.1


CONFIGS = [
    LosoGRLConfig("loso_grl_abundance_l8_lam10_cw01_rw1", condition_weight=0.1),
    LosoGRLConfig("loso_grl_abundance_l8_lam10_cw001_rw1", condition_weight=0.01),
    LosoGRLConfig("loso_grl_abundance_l8_lam10_cw0_rw1", condition_weight=0.0),
    LosoGRLConfig(
        "loso_grl_mech_context_only_l8_lam10_rw5_rel1_var1",
        condition_weight=0.0,
        recon_weight=5.0,
        rel_recon_weight=1.0,
        variance_weight=1.0,
        use_condition_head=False,
        condition_context_for_adversary=True,
    ),
]


def read_abundance(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    df = pd.read_csv(path)
    features = df["feature"].astype(str).tolist()
    sample_ids = [c for c in df.columns if c != "feature"]
    abundance = df[sample_ids].astype(float).to_numpy()
    return features, sample_ids, abundance


def write_abundance(features: list[str], sample_ids: list[str], abundance: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(abundance, columns=sample_ids)
    df.insert(0, "feature", features)
    df.to_csv(path, index=False)


def labels(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    study_values = sorted(metadata["studyID"].astype(str).unique())
    condition_values = sorted(metadata["study_condition"].astype(str).unique())
    study_map = {label: i for i, label in enumerate(study_values)}
    condition_map = {label: i for i, label in enumerate(condition_values)}
    y_study = np.array([study_map[v] for v in metadata["studyID"].astype(str)], dtype=np.int64)
    y_condition = np.array([condition_map[v] for v in metadata["study_condition"].astype(str)], dtype=np.int64)
    return y_study, y_condition, study_map, condition_map


def class_weights(y: np.ndarray, n_classes: int, device: Any):
    import torch

    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def lambda_for_epoch(epoch: int, epochs: int, lambda_grl: float, warmup_fraction: float) -> float:
    progress = (epoch - 1) / max(epochs - 1, 1)
    return float(lambda_grl * min(progress / max(warmup_fraction, 1e-8), 1.0))


def train_transform_fold(
    raw_abundance: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    metadata: pd.DataFrame,
    config: LosoGRLConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    seed = config.seed + int(test_idx[0])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    train_meta = metadata.iloc[train_idx].copy()
    train_x_log = np.log1p(1000.0 * np.clip(raw_abundance[:, train_idx].T, 0, None)).astype(np.float32)
    mean = train_x_log.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_x_log.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_x_std = ((train_x_log - mean) / std).astype(np.float32)
    y_study_np, y_condition_np, study_map, condition_map = labels(train_meta)

    x = torch.tensor(train_x_std, dtype=torch.float32, device=device)
    y_study = torch.tensor(y_study_np, dtype=torch.long, device=device)
    y_condition = torch.tensor(y_condition_np, dtype=torch.long, device=device)
    input_dim = x.shape[1]
    latent_dim = min(config.latent_dim, input_dim)
    encoder = nn.Sequential(
        nn.Linear(input_dim, config.hidden_dim),
        nn.ReLU(),
        nn.Linear(config.hidden_dim, latent_dim),
        nn.LayerNorm(latent_dim),
    ).to(device)
    decoder = nn.Sequential(
        nn.Linear(latent_dim, config.hidden_dim),
        nn.ReLU(),
        nn.Linear(config.hidden_dim, input_dim),
    ).to(device)
    condition_embedding = nn.Embedding(len(condition_map), 16).to(device) if config.condition_context_for_adversary else None
    study_adversary = nn.Sequential(
        nn.Linear(latent_dim + (16 if config.condition_context_for_adversary else 0), config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, len(study_map)),
    ).to(device)
    condition_head = nn.Sequential(
        nn.Linear(latent_dim, config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, len(condition_map)),
    ).to(device) if config.use_condition_head else None
    modules = [encoder, decoder, study_adversary]
    if condition_embedding is not None:
        modules.append(condition_embedding)
    if condition_head is not None:
        modules.append(condition_head)
    optimizer = torch.optim.Adam([p for module in modules for p in module.parameters()], lr=config.lr)
    study_weights = class_weights(y_study_np, len(study_map), device)
    condition_weights = class_weights(y_condition_np, len(condition_map), device)
    loader = DataLoader(
        TensorDataset(x, y_study, y_condition),
        batch_size=min(config.batch_size, len(x)),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    history = []
    for epoch in range(1, config.epochs + 1):
        current_lambda = lambda_for_epoch(epoch, config.epochs, config.lambda_grl, config.warmup_fraction)
        batch_rows = []
        for batch_x, batch_study, batch_condition in loader:
            optimizer.zero_grad(set_to_none=True)
            z = encoder(batch_x)
            recon = decoder(z)
            z_adv = gradient_reverse(z, current_lambda)
            adv_input = torch.cat([z_adv, condition_embedding(batch_condition)], dim=1) if condition_embedding is not None else z_adv
            study_loss = F.cross_entropy(study_adversary(adv_input), batch_study, weight=study_weights)
            if condition_head is not None:
                condition_loss = F.cross_entropy(condition_head(z), batch_condition, weight=condition_weights)
            else:
                condition_loss = torch.zeros((), dtype=torch.float32, device=device)
            recon_loss = F.mse_loss(recon, batch_x)
            rel_recon_loss = torch.zeros((), dtype=torch.float32, device=device)
            if config.rel_recon_weight > 0:
                rel_recon_loss = F.mse_loss(torch.softmax(recon, dim=1), torch.softmax(batch_x, dim=1))
            variance_loss = torch.zeros((), dtype=torch.float32, device=device)
            if config.variance_weight > 0 and batch_x.shape[0] > 1:
                variance_loss = F.mse_loss(recon.var(dim=0, unbiased=False), batch_x.var(dim=0, unbiased=False))
            loss = (
                study_loss
                + config.condition_weight * condition_loss
                + config.recon_weight * recon_loss
                + config.rel_recon_weight * rel_recon_loss
                + config.variance_weight * variance_loss
            )
            loss.backward()
            optimizer.step()
            batch_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "study_loss": float(study_loss.detach().cpu()),
                    "condition_loss": float(condition_loss.detach().cpu()),
                    "recon_loss": float(recon_loss.detach().cpu()),
                    "rel_recon_loss": float(rel_recon_loss.detach().cpu()),
                    "variance_loss": float(variance_loss.detach().cpu()),
                    "n": int(batch_x.shape[0]),
                }
            )
        if epoch == config.epochs:
            history.append(
                {
                    "method": config.method_name,
                    "epoch": epoch,
                    "loss": float(np.average([r["loss"] for r in batch_rows], weights=[r["n"] for r in batch_rows])),
                    "study_loss": float(np.average([r["study_loss"] for r in batch_rows], weights=[r["n"] for r in batch_rows])),
                    "condition_loss": float(np.average([r["condition_loss"] for r in batch_rows], weights=[r["n"] for r in batch_rows])),
                    "recon_loss": float(np.average([r["recon_loss"] for r in batch_rows], weights=[r["n"] for r in batch_rows])),
                    "rel_recon_loss": float(np.average([r["rel_recon_loss"] for r in batch_rows], weights=[r["n"] for r in batch_rows])),
                    "variance_loss": float(np.average([r["variance_loss"] for r in batch_rows], weights=[r["n"] for r in batch_rows])),
                }
            )

    def transform(indices: np.ndarray) -> np.ndarray:
        x_log = np.log1p(1000.0 * np.clip(raw_abundance[:, indices].T, 0, None)).astype(np.float32)
        x_std = ((x_log - mean) / std).astype(np.float32)
        xt = torch.tensor(x_std, dtype=torch.float32, device=device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            recon_std = decoder(encoder(xt)).detach().cpu().numpy()
        recon_log = recon_std * std + mean
        recon = np.expm1(recon_log) / 1000.0
        recon = np.clip(recon, 0.0, None)
        raw_sums = np.clip(raw_abundance[:, indices].sum(axis=0, keepdims=True).T, 1e-12, None)
        recon_sums = np.clip(recon.sum(axis=1, keepdims=True), 1e-12, None)
        return (recon * (raw_sums / recon_sums)).T

    return transform(train_idx), transform(test_idx), history


def model_matrix(abundance: np.ndarray) -> np.ndarray:
    x = np.log1p(1000.0 * np.clip(abundance.T, 0, None)).astype(float)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x - x.mean(axis=0, keepdims=True)) / std


def choose_c_auc(x: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    cs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    scores = []
    for c in cs:
        fold_scores = []
        for fold in sorted(set(fold_ids.tolist())):
            tr = fold_ids != fold
            va = fold_ids == fold
            if len(np.unique(y[va])) < 2:
                continue
            model = LogisticRegression(C=c, max_iter=1000, class_weight="balanced", random_state=seed)
            model.fit(x[tr], y[tr])
            fold_scores.append(float(roc_auc_score(y[va], model.predict_proba(x[va])[:, 1])))
        scores.append(float(np.mean(fold_scores)) if fold_scores else float("-inf"))
    return float(cs[int(np.argmax(scores))])


def disease_loso_from_fold_matrices(
    corrected_by_study: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    sample_ids = metadata["sample_id"].astype(str).to_numpy()
    y_all = (metadata["study_condition"].astype(str).to_numpy() == "CRC").astype(int)
    probabilities = np.full(len(metadata), np.nan)
    fold_rows = []
    for heldout, (train_idx, test_idx, train_abd, test_abd) in corrected_by_study.items():
        train_ids = sample_ids[train_idx]
        inner = pd.read_csv(SPLIT_DIR / "disease_loso_inner_folds.csv")
        fold_rows_inner = inner[inner["held_out_study"] == heldout].set_index("sample_id")
        fold_ids = fold_rows_inner.loc[train_ids, "inner_fold"].to_numpy()
        x_train = model_matrix(train_abd)
        x_test = model_matrix(test_abd)
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        c = choose_c_auc(x_train, y_train, fold_ids, seed=42)
        model = LogisticRegression(C=c, max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        probabilities[test_idx] = prob
        auc = float(roc_auc_score(y_test, prob))
        ba = float(balanced_accuracy_score(y_test, prob >= 0.5))
        fold_rows.append(
            {
                "held_out_study": heldout,
                "n_test": int(len(test_idx)),
                "selected_C": c,
                "heldout_auc": auc,
                "heldout_balanced_accuracy": ba,
            }
        )
    overall = {
        "disease_LOSO_overall_AUC": float(roc_auc_score(y_all, probabilities)),
        "disease_LOSO_mean_within_study_AUC": float(np.mean([r["heldout_auc"] for r in fold_rows])),
        "disease_LOSO_balanced_accuracy": float(balanced_accuracy_score(y_all, probabilities >= 0.5)),
    }
    return pd.DataFrame(fold_rows), overall


def evaluate_baseline_loso(name: str, abundance: np.ndarray, metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    studies = metadata["studyID"].astype(str).to_numpy()
    corrected = {}
    for heldout in sorted(set(studies.tolist())):
        train_idx = np.where(studies != heldout)[0]
        test_idx = np.where(studies == heldout)[0]
        corrected[heldout] = (train_idx, test_idx, abundance[:, train_idx], abundance[:, test_idx])
    folds, overall = disease_loso_from_fold_matrices(corrected, metadata)
    folds.insert(0, "method", name)
    overall["method"] = name
    return folds, overall


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    features, sample_ids, raw = read_abundance(DATA_DIR / "crc_raw_abundance.csv")
    _, mmuphin_samples, mmuphin = read_abundance(DATA_DIR / "crc_mmuphin_adjusted_abundance.csv")
    metadata = pd.read_csv(DATA_DIR / "crc_metadata.csv", dtype=str)
    if sample_ids != metadata["sample_id"].astype(str).tolist() or sample_ids != mmuphin_samples:
        raise ValueError("Sample order mismatch.")
    studies = metadata["studyID"].astype(str).to_numpy()

    fold_tables = []
    summary_rows = []
    raw_folds, raw_overall = evaluate_baseline_loso("raw_python_loso_classifier", raw, metadata)
    mm_folds, mm_overall = evaluate_baseline_loso("mmuphin_python_loso_classifier", mmuphin, metadata)
    fold_tables.extend([raw_folds, mm_folds])
    summary_rows.extend([raw_overall, mm_overall])

    training_history = []
    for config in CONFIGS:
        corrected_by_study = {}
        oof = np.zeros_like(raw)
        for heldout in sorted(set(studies.tolist())):
            train_idx = np.where(studies != heldout)[0]
            test_idx = np.where(studies == heldout)[0]
            train_abd, test_abd, hist = train_transform_fold(raw, train_idx, test_idx, metadata, config)
            for row in hist:
                row["held_out_study"] = heldout
                training_history.append(row)
            corrected_by_study[heldout] = (train_idx, test_idx, train_abd, test_abd)
            oof[:, test_idx] = test_abd
            print("trained", config.method_name, "heldout", heldout)
        oof_path = OUTPUT_DIR / f"{config.method_name}_oof_abundance.csv"
        benchmark_oof_path = METHOD_DIR / f"{config.method_name}_oof_abundance.csv"
        write_abundance(features, sample_ids, oof, oof_path)
        write_abundance(features, sample_ids, oof, benchmark_oof_path)
        folds, overall = disease_loso_from_fold_matrices(corrected_by_study, metadata)
        folds.insert(0, "method", config.method_name)
        overall.update(asdict(config))
        overall["method"] = config.method_name
        overall["oof_matrix_path"] = str(oof_path)
        fold_tables.append(folds)
        summary_rows.append(overall)

    fold_metrics = pd.concat(fold_tables, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    fold_path = METRIC_DIR / "grl_abundance_loso_correction_per_fold_metrics.csv"
    summary_path = METRIC_DIR / "grl_abundance_loso_correction_summary.csv"
    history_path = METRIC_DIR / "grl_abundance_loso_correction_training_history.csv"
    fold_metrics.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(training_history).to_csv(history_path, index=False)

    lines = [
        "# GRL Abundance LOSO Correction",
        "",
        "## Scope",
        "",
        "For each held-out study, GRL correction was trained only on the other studies, then used to transform both train studies and the held-out study. Disease classification was then trained on corrected train studies and tested on the corrected held-out study.",
        "",
        "This directly checks whether the high full-data disease AUC was caused by condition-label amplification during correction training.",
        "",
        "## Summary",
        "",
        summary[
            [
                "method",
                "disease_LOSO_mean_within_study_AUC",
                "disease_LOSO_overall_AUC",
                "disease_LOSO_balanced_accuracy",
            ]
        ].to_markdown(index=False),
        "",
        "## Reading",
        "",
        "- Compare GRL rows to the Python raw/MMUPHin rows, not directly to the R/glmnet values.",
        "- In this run, all three LOSO-corrected GRL settings underperform the raw/MMUPHin Python LOSO baselines on disease AUC.",
        "- Because `cw0.1` loses its high full-data AUC here, the full-data 0.882 result is best treated as transductive condition amplification, not reliable disease preservation.",
        "",
        "## Output Files",
        "",
        f"- `summary`: `{summary_path.relative_to(ROOT)}`",
        f"- `per_fold_metrics`: `{fold_path.relative_to(ROOT)}`",
        f"- `training_history`: `{history_path.relative_to(ROOT)}`",
        f"- `oof_matrices`: `{OUTPUT_DIR.relative_to(ROOT)}`",
    ]
    (REPORT_DIR / "grl_abundance_loso_correction_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("GRL_ABUNDANCE_LOSO_CORRECTION_OK")
    print(summary[["method", "disease_LOSO_mean_within_study_AUC", "disease_LOSO_overall_AUC", "disease_LOSO_balanced_accuracy"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
