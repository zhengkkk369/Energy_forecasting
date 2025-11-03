from __future__ import annotations

import torch
from torch import nn


class TS2VecForecaster(nn.Module):
    """Simplified TS2Vec-style encoder combining convolution and GRU."""

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        output_dim: int = 1,
        hidden_dim: int = 128,
        rep_dim: int = 128,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=rep_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            nn.GELU(),
            nn.Linear(rep_dim, pred_len * output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.transpose(1, 2)
        features = torch.relu(self.conv1(x_perm))
        features = self.norm(torch.relu(self.conv2(features)))
        features = features.transpose(1, 2)
        _, hidden = self.encoder(features)
        embedding = hidden[-1]
        out = self.head(embedding)
        return out.view(x.size(0), self.pred_len, self.output_dim)


def build_ts2vec(
    input_dim: int,
    pred_len: int,
    output_dim: int,
    hidden_dim: int,
    rep_dim: int,
) -> nn.Module:
    return TS2VecForecaster(
        input_dim=input_dim,
        pred_len=pred_len,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        rep_dim=rep_dim,
    )
