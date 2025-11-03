from __future__ import annotations

import torch
from torch import nn


class DLinearForecaster(nn.Module):
    """Implementation of the DLinear idea using seasonal/trend linear projections."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.seasonal_layer = nn.Linear(seq_len, pred_len)
        self.trend_layer = nn.Linear(seq_len, pred_len)
        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        seasonal = self.seasonal_layer(x_t)
        trend = self.trend_layer(x_t)
        combined = seasonal + trend
        combined = combined.transpose(1, 2)
        out = self.output_proj(combined)
        return out.view(x.size(0), self.pred_len, self.output_dim)


def build_dlinear(input_dim: int, seq_len: int, pred_len: int, output_dim: int) -> nn.Module:
    return DLinearForecaster(
        input_dim=input_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        output_dim=output_dim,
    )
