from __future__ import annotations


class BiomeGPTModelUnavailable(RuntimeError):
    pass


def build_model_from_checkpoint(_checkpoint):
    """Placeholder for the original BiomeGPT architecture.

    The old package contains training and pipeline scripts, but no verified
    minimal model constructor/checkpoint contract for this CRC benchmark. This
    function deliberately fails clearly until a current checkpoint and matching
    model definition are supplied.
    """
    raise BiomeGPTModelUnavailable(
        "BiomeGPT model reconstruction is not wired yet. Provide the current "
        "architecture file and checkpoint contract before running CLS extraction."
    )

