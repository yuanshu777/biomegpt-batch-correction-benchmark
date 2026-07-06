from __future__ import annotations


def cross_entropy_loss(logits, labels):
    import torch.nn.functional as F

    return F.cross_entropy(logits, labels)


def reconstruction_mse(predicted, target):
    import torch.nn.functional as F

    return F.mse_loss(predicted, target)

