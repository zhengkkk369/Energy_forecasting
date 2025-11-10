"""Building blocks used by the simplified OneNet model."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ChannelMixing(nn.Module):
    """Simple feed-forward channel mixing with residual connection."""

    def __init__(self, hidden_dim: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.activation = _get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + residual
        return self.norm(x)


class MultiScaleConvBlock(nn.Module):
    """Multi-branch temporal convolution block.

    The block applies a set of convolutions with different kernel sizes and
    aggregates their responses.  A light-weight channel mixing network adds
    further capacity while keeping the module inexpensive.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        kernel_sizes: Iterable[int],
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        kernel_sizes = tuple(kernel_sizes)
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one value")

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=ks,
                    padding=ks // 2,
                    groups=hidden_dim,
                    bias=False,
                )
                for ks in kernel_sizes
            ]
        )
        self.pointwise = nn.Conv1d(hidden_dim * len(kernel_sizes), hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.activation = _get_activation(activation)
        self.channel_mixing = ChannelMixing(hidden_dim, dropout, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        outputs = [conv(x) for conv in self.convs]
        x = torch.cat(outputs, dim=1)
        x = self.pointwise(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x)

        # Channel mixing expects shape (batch, seq_len, channels)
        x = x.transpose(1, 2)
        x = self.channel_mixing(x)
        x = x.transpose(1, 2)
        return x
