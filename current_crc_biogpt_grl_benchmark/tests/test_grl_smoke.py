import json

import numpy as np
import pandas as pd
import pytest


def make_synthetic_crc_embeddings():
    rng = np.random.default_rng(7)
    n_samples = 60
    n_dim = 16
    sample_ids = [f"s{i:03d}" for i in range(n_samples)]
    studies = np.array(["study_a"] * 20 + ["study_b"] * 20 + ["study_c"] * 20)
    conditions = np.array(["CRC", "control"] * 30)
    x = rng.normal(size=(n_samples, n_dim)).astype(float)
    x[studies == "study_b", 0] += 1.5
    x[studies == "study_c", 1] -= 1.5
    x[conditions == "CRC", 2] += 0.75
    embeddings = pd.DataFrame(x, columns=[f"dim_{i}" for i in range(n_dim)])
    embeddings.insert(0, "sample_id", sample_ids)
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "studyID": studies,
            "study_condition": conditions,
        }
    )
    return embeddings, metadata


def test_grl_embedding_smoke_and_save(tmp_path):
    pytest.importorskip("torch")
    from src.grl_correction.train_grl import save_grl_result, train_embedding_grl

    embeddings, metadata = make_synthetic_crc_embeddings()
    result = train_embedding_grl(
        embeddings,
        metadata,
        epochs=3,
        hidden_dim=16,
        latent_dim=16,
        batch_size=12,
        preserve_weight=0.5,
        lambda_schedule="linear",
        warmup_fraction=0.5,
        condition_aware_adversary=True,
        external_eval_every=2,
        seed=11,
    )
    assert result.corrected_embeddings.shape == (60, 17)
    assert list(result.corrected_embeddings.columns)[0] == "sample_id"
    assert len(result.history) == 3
    assert "preservation_loss" in result.history.columns
    assert not result.final_probe_metrics.empty

    paths = save_grl_result(result, tmp_path)
    for path in paths.values():
        assert path
    assert (tmp_path / "biogpt_grl_corrected_cls_389.csv").exists()
    assert (tmp_path / "grl_training_history.csv").exists()
    label_maps = json.loads((tmp_path / "grl_label_maps.json").read_text())
    assert set(label_maps) == {"study_labels", "condition_labels"}


def test_grl_study_conditioned_decoder_smoke():
    pytest.importorskip("torch")
    from src.grl_correction.train_grl import train_embedding_grl

    embeddings, metadata = make_synthetic_crc_embeddings()
    result = train_embedding_grl(
        embeddings,
        metadata,
        epochs=2,
        hidden_dim=16,
        latent_dim=8,
        batch_size=15,
        preserve_weight=0.25,
        use_study_conditioned_decoder=True,
        lambda_schedule="constant",
        external_eval_every=None,
    )
    assert result.corrected_embeddings.shape == (60, 9)
    assert len(result.history) == 2
    assert "preservation_loss" in result.history.columns


def test_grl_validation_early_stopping_smoke():
    pytest.importorskip("torch")
    from src.grl_correction.crossfit import EarlyStoppingConfig, train_grl_with_validation_early_stopping

    embeddings, metadata = make_synthetic_crc_embeddings()
    val_idx = [0, 1, 20, 21, 40, 41, 2, 22, 42, 3, 23, 43]
    transform_idx = [4, 5, 24, 25, 44, 45]
    train_idx = [i for i in range(len(metadata)) if i not in set(val_idx + transform_idx)]
    train_meta = metadata.iloc[train_idx].copy()
    val_meta = metadata.iloc[val_idx].copy()
    transform_meta = metadata.iloc[transform_idx].copy()
    train_x = embeddings[embeddings["sample_id"].isin(train_meta["sample_id"])].copy()
    val_x = embeddings[embeddings["sample_id"].isin(val_meta["sample_id"])].copy()
    transform_x = embeddings[embeddings["sample_id"].isin(transform_meta["sample_id"])].copy()

    result = train_grl_with_validation_early_stopping(
        train_x,
        train_meta,
        val_x,
        val_meta,
        transform_x,
        config=EarlyStoppingConfig(
            latent_dim=4,
            hidden_dim=12,
            epochs=5,
            eval_every=1,
            patience_evals=2,
            batch_size=12,
            condition_weight=0.1,
            preserve_weight=0.001,
            seed=5,
        ),
        fold_id="smoke",
    )
    assert result.corrected_embeddings.shape == (6, 5)
    assert not result.trace.empty
    assert result.selected["fold_id"] == "smoke"


def test_nomean_cls_adapter_smoke():
    pytest.importorskip("torch")
    from src.grl_correction.nomean_adapter import NoMeanAdapterConfig, train_nomean_cls_adapter

    embeddings, metadata = make_synthetic_crc_embeddings()
    result = train_nomean_cls_adapter(
        embeddings,
        metadata,
        config=NoMeanAdapterConfig(
            hidden_dim=12,
            study_embedding_dim=4,
            epochs=3,
            warmup_epochs=1,
            batch_size=15,
            lambda_grl=0.05,
            adversary_weight=0.05,
            seed=13,
        ),
    )
    assert result.corrected_embeddings.shape == embeddings.shape
    assert len(result.training_history) == 3
    assert "effective_rank" in result.diagnostics
    assert set(result.label_maps) == {"study_labels", "condition_labels"}


def test_split_cls_adapter_smoke():
    pytest.importorskip("torch")
    from src.grl_correction.split_adapter import SplitAdapterConfig, train_split_cls_adapter

    embeddings, metadata = make_synthetic_crc_embeddings()
    result = train_split_cls_adapter(
        embeddings,
        metadata,
        config=SplitAdapterConfig(
            inv_dim=8,
            nuisance_dim=4,
            hidden_dim=16,
            epochs=3,
            warmup_epochs=1,
            batch_size=15,
            lambda_grl=0.05,
            adversary_weight=0.05,
            seed=17,
        ),
    )
    assert result.corrected_embeddings.shape == (60, 9)
    assert result.nuisance_embeddings.shape == (60, 5)
    assert len(result.training_history) == 3
    assert "effective_rank" in result.diagnostics
