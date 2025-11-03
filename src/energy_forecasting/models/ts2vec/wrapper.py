from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import nn

from . import encoder as base_encoder


class TS2VecRegressor(nn.Module):
    """Generic wrapper turning a TS2Vec encoder into a forecaster."""

    def __init__(
        self,
        encoder_factory: Callable[..., nn.Module],
        input_dim: int,
        pred_len: int,
        output_dim: int,
        rep_dim: int = 320,
        hidden_dim: int = 64,
        depth: int = 10,
        extra_kwargs: Dict | None = None,
    ) -> None:
        super().__init__()
        extra_kwargs = extra_kwargs or {}
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.encoder = encoder_factory(
            input_dims=input_dim,
            output_dims=rep_dim,
            hidden_dims=hidden_dim,
            depth=depth,
            **extra_kwargs,
        )
        self.regressor = nn.Linear(rep_dim, pred_len * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "device"):
            self.encoder.device = x.device
        representation = self.encoder(x, mask="all_true")[:, -1]
        out = self.regressor(representation)
        return out.view(x.size(0), self.pred_len, self.output_dim)


def build_fsnet_model(
    input_dim: int,
    pred_len: int,
    output_dim: int,
    rep_dim: int,
    hidden_dim: int,
    depth: int,
    gamma: float,
) -> nn.Module:
    return TS2VecRegressor(
        encoder_factory=base_encoder.TSEncoder,
        input_dim=input_dim,
        pred_len=pred_len,
        output_dim=output_dim,
        rep_dim=rep_dim,
        hidden_dim=hidden_dim,
        depth=depth,
    )


def build_nomem_model(
    input_dim: int,
    pred_len: int,
    output_dim: int,
    rep_dim: int,
    hidden_dim: int,
    depth: int,
    gamma: float,
) -> nn.Module:
    return TS2VecRegressor(
        encoder_factory=base_encoder.TSEncoder,
        input_dim=input_dim,
        pred_len=pred_len,
        output_dim=output_dim,
        rep_dim=rep_dim,
        hidden_dim=hidden_dim,
        depth=depth,
    )


def build_ncca_model(
    input_dim: int,
    pred_len: int,
    output_dim: int,
    rep_dim: int,
    hidden_dim: int,
    depth: int,
    gamma: float,
) -> nn.Module:
    return TS2VecRegressor(
        encoder_factory=base_encoder.TSEncoder,
        input_dim=input_dim,
        pred_len=pred_len,
        output_dim=output_dim,
        rep_dim=rep_dim,
        hidden_dim=hidden_dim,
        depth=depth,
    )
