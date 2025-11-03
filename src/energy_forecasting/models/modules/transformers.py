from __future__ import annotations

import torch
from torch import nn

from .common import PositionalEncoding


class TransformerForecaster(nn.Module):
    """Transformer encoder with positional encoding and pooled head."""

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        output_dim: int = 1,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = pooling
        self.head = nn.Linear(d_model, pred_len * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x)
        encoded = self.encoder(self.positional_encoding(projected))
        if self.pooling == "mean":
            pooled = encoded.mean(dim=1)
        else:
            pooled = encoded[:, -1]
        out = self.head(pooled)
        return out.view(x.size(0), self.pred_len, self.output_dim)


def build_transformer(
    input_dim: int,
    pred_len: int,
    output_dim: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
    pooling: str,
) -> nn.Module:
    return TransformerForecaster(
        input_dim=input_dim,
        pred_len=pred_len,
        output_dim=output_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        pooling=pooling,
    )


class InformerForecaster(nn.Module):
    """Simplified Informer architecture using probabilistic attention masks."""

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        output_dim: int = 1,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        factor: int = 5,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.factor = factor
        self.projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, pred_len * output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        mask = self._build_mask(x)
        encoded = self.encoder(self.positional_encoding(projected), mask=mask)
        out = self.head(encoded[:, -1])
        return out.view(x.size(0), self.pred_len, self.output_dim)

    def _build_mask(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        keep = max(1, seq_len // self.factor)
        mask = torch.zeros(seq_len, seq_len, device=x.device)
        if keep < seq_len:
            mask[:, : seq_len - keep] = float("-inf")
        return mask


def build_informer(
    input_dim: int,
    pred_len: int,
    output_dim: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
) -> nn.Module:
    return InformerForecaster(
        input_dim=input_dim,
        pred_len=pred_len,
        output_dim=output_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
    )
