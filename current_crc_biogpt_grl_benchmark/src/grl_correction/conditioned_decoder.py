from __future__ import annotations


def _nn():
    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for conditioned decoder modules.") from exc
    return nn


class StudyConditionedDecoder(_nn().Module):
    def __init__(self, latent_dim: int, n_studies: int, output_dim: int, hidden_dim: int = 128):
        nn = _nn()
        super().__init__()
        self.study_embedding = nn.Embedding(n_studies, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z, study_index):
        import torch

        study = self.study_embedding(study_index)
        return self.net(torch.cat([z, study], dim=1))
