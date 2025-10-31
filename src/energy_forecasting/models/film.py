from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class FiLMConditioner(nn.Module):
    """Maps context embeddings to FiLM modulation parameters."""

    def __init__(self, d_in: int, d_model: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(d_in, d_model)
        self.beta = nn.Linear(d_in, d_model)

    def forward(self, context: Dict[str, torch.Tensor], hidden: torch.Tensor) -> torch.Tensor:
        embedding = torch.cat([v for v in context.values()], dim=-1)
        gamma = self.gamma(embedding)
        beta = self.beta(embedding)
        return gamma * hidden + beta
