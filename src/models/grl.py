"""Gradient reversal layer.

Identity on the forward pass; multiplies the incoming gradient by -lambda on
the way back. That is the whole of DANN's adversary: the domain head minimises
domain loss, the trunk maximises it, and both run in one backward pass.

lambda is a Python float captured in the autograd context, not a tensor, so it
does not itself receive a gradient.
"""

import math

import torch


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = float(lam)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg().mul(ctx.lam), None


def grl(x, lam):
    """Reverse the gradient on `x`, scaled by `lam`."""
    return _Reverse.apply(x, lam)


def lambda_at(step, n_steps, gamma=10.0, lam_max=1.0):
    """Ganin schedule: 0 at the start of training, `lam_max` at the end.

    lambda_p = 2 / (1 + exp(-gamma * p)) - 1, p in [0, 1]. Starting at full
    strength dumps whatever the domain head can see, including PD.
    """
    if n_steps <= 1:
        return float(lam_max)
    p = min(max(step / (n_steps - 1), 0.0), 1.0)
    return float(lam_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0))
