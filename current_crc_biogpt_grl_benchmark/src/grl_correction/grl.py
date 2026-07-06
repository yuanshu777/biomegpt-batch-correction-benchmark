from __future__ import annotations


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for GRL training.") from exc
    return torch


class GradientReversalFunction:
    @staticmethod
    def apply(x, lambda_):
        torch = _torch()

        class _GRL(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor, scale):
                ctx.scale = scale
                return input_tensor.view_as(input_tensor)

            @staticmethod
            def backward(ctx, grad_output):
                return -ctx.scale * grad_output, None

        return _GRL.apply(x, lambda_)


def gradient_reverse(x, lambda_: float = 1.0):
    return GradientReversalFunction.apply(x, lambda_)

