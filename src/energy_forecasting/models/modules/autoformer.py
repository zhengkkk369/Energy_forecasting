from __future__ import annotations

import torch
from torch import nn


class AutoformerForecaster(nn.Module):
    """Lightweight Autoformer-inspired model with series decomposition."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        output_dim: int = 1,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.moving_avg = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.encoder_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.seasonal_head = nn.Linear(d_model, pred_len * output_dim)
        self.trend_head = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        encoded = self.encoder(self.encoder_proj(seasonal))
        seasonal_out = self.seasonal_head(encoded[:, -1]).view(x.size(0), self.pred_len, self.output_dim)
        trend_component = self.trend_head(trend.mean(dim=-1))
        trend_out = trend_component.unsqueeze(-1).repeat(1, 1, self.output_dim)
        return seasonal_out + trend_out


def build_autoformer(
    input_dim: int,
    seq_len: int,
    pred_len: int,
    output_dim: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    ff_dim: int,
) -> nn.Module:
    return AutoformerForecaster(
        input_dim=input_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        output_dim=output_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
    )
