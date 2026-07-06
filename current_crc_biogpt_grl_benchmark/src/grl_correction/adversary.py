from __future__ import annotations


def _nn():
    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for adversary modules.") from exc
    return nn


class MLPAdversary(_nn().Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 128, dropout: float = 0.1):
        nn = _nn()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)

