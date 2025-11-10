"""Simplified OneNet model definition.

The original OneNet implementation provides a highly optimised set of
multi-branch convolutional blocks.  This port keeps the overall idea but
reimplements the layers with lightweight PyTorch modules so that it is
self-contained and easy to extend inside this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from .layers import MultiScaleConvBlock


@dataclass
class OneNetConfig:
    """Configuration for the :class:`OneNet` model.

    Parameters are intentionally close to the public OneNet repository so the
    model can be configured with familiar names while remaining fully
    compatible with the project's :class:`~energy_forecasting.models.config.ModelConfig`.
    """

    input_dim: int
    output_dim: int
    seq_len: int
    pred_len: int
    d_model: int
    num_layers: int
    dropout: float
    kernel_sizes: Sequence[int]
    activation: str = "gelu"


class OneNet(nn.Module):
    """A compact implementation of the OneNet forecasting model.

    The network first projects the multivariate time series into a latent
    representation.  A stack of :class:`MultiScaleConvBlock` modules performs
    channel-mixing and temporal aggregation using different receptive fields.
    Finally, an adaptive pooling layer compresses the latent sequence into the
    desired prediction horizon and a linear projection recovers the target
    dimensionality.
    """

    def __init__(self, cfg: OneNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_projection = nn.Linear(cfg.input_dim, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                MultiScaleConvBlock(
                    hidden_dim=cfg.d_model,
                    kernel_sizes=cfg.kernel_sizes,
                    dropout=cfg.dropout,
                    activation=cfg.activation,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(cfg.pred_len)
        self.output_projection = nn.Linear(cfg.d_model, cfg.output_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast future values.

        Parameters
        ----------
        x:
            Tensor with shape ``(batch, seq_len, input_dim)``.

        Returns
        -------
        torch.Tensor
            Forecast with shape ``(batch, pred_len, output_dim)``.
        """

        if x.dim() != 3:
            raise ValueError(
                "Expected input tensor with shape (batch, seq_len, input_dim), "
                f"but received {tuple(x.shape)}"
            )

        if x.size(1) != self.cfg.seq_len:
            raise ValueError(
                "Sequence length mismatch: configured seq_len="
                f"{self.cfg.seq_len} but received {x.size(1)}"
            )

        x = self.input_projection(x)
        # Move channels to the convolutional dimension: (batch, channels, seq_len)
        x = x.transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        x = self.pool(x)
        x = x.transpose(1, 2)
        x = self.output_projection(x)
        return x


def build_onenet(
    *,
    input_dim: int,
    output_dim: int,
    seq_len: int,
    pred_len: int,
    d_model: int,
    num_layers: int,
    dropout: float,
    kernel_sizes: Iterable[int],
    activation: str = "gelu",
) -> OneNet:
    """Factory helper compatible with the model registry."""

    cfg = OneNetConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
        kernel_sizes=tuple(kernel_sizes),
        activation=activation,
    )
    return OneNet(cfg)
