"""Residual wrapper: predict a correction on top of a precomputed base surface."""

import torch
import torch.nn as nn


class ResidualSurfaceModel(nn.Module):
    """``prediction = base + net(inputs)`` where ``base`` is the last ``n_quotes`` input columns.

    The final layer of ``net`` is zero-initialised, so an untrained model returns
    the base surface exactly and training can only move away from it if that
    lowers the loss. The network sees the base as an input feature as well.
    """

    def __init__(self, net: nn.Module, n_quotes: int, zero_init: bool = True) -> None:
        super().__init__()
        self.net = net
        self.n_quotes = n_quotes
        if zero_init:
            last = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -self.n_quotes :] + self.net(x)
