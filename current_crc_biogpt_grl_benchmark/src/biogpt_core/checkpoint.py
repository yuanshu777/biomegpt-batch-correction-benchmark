from __future__ import annotations

from pathlib import Path
from typing import Any


class MissingAssetError(FileNotFoundError):
    """Raised when a required model/data asset is not configured or absent."""


def require_asset(path: str | Path | None, name: str) -> Path:
    if path is None or str(path).strip().lower() in {"", "none", "null"}:
        raise MissingAssetError(f"{name} is not configured.")
    resolved = Path(path)
    if not resolved.exists():
        raise MissingAssetError(f"{name} is missing: {resolved}")
    return resolved


def load_torch_checkpoint(path: str | Path | None) -> Any:
    checkpoint_path = require_asset(path, "BiomeGPT checkpoint")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to load a BiomeGPT checkpoint.") from exc
    return torch.load(checkpoint_path, map_location="cpu")

