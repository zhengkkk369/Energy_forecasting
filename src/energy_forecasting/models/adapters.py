from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


class LayerAdapter(nn.Module):
    """Bottle-neck adapter module."""

    def __init__(self, d_model: int, bottleneck_ratio: float = 0.25) -> None:
        super().__init__()
        hidden = max(1, int(d_model * bottleneck_ratio))
        self.down = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


@dataclass
class AdapterState:
    params: Dict[str, torch.Tensor]

    def clone(self) -> "AdapterState":
        return AdapterState({k: v.clone() for k, v in self.params.items()})
