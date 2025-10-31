from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


class TemporalTransformer(nn.Module):
    """Minimal transformer encoder placeholder."""

    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        return self.output(hidden)


class TemporalConvNet(nn.Module):
    """Placeholder TCN backbone for experimentation."""

    def __init__(self, input_size: int, hidden_size: int = 64, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.act = nn.GELU()
        self.head = nn.Conv1d(hidden_size, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects x with shape [batch, time, features]
        x = x.transpose(1, 2)
        hidden = self.act(self.conv(x))
        out = self.head(hidden)
        return out.transpose(1, 2)


@dataclass
class BackboneFactory:
    name: str
    params: Dict[str, int]

    def build(self) -> nn.Module:
        if self.name == "temporal_transformer":
            return TemporalTransformer(**self.params)
        if self.name == "tcn":
            return TemporalConvNet(**self.params)
        raise ValueError(f"Unknown backbone: {self.name}")
