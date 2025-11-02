from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Simple encoder-only LSTM forecaster followed by a linear projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 24,
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_dim, pred_len * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        _, (hidden, _) = self.encoder(x)
        representation = hidden[-1]  # [batch, hidden_dim]
        out = self.projection(representation)
        return out.view(x.size(0), self.pred_len, self.output_dim)


@dataclass
class BaselineConfig:
    input_dim: int
    output_dim: int
    pred_len: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    def build(self) -> nn.Module:
        return LSTMForecaster(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            pred_len=self.pred_len,
            output_dim=self.output_dim,
            dropout=self.dropout,
        )
