from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    """Chomp layer trims convolutional padding from the right side."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual temporal convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.dropout1(self.activation(self.chomp1(y)))
        y = self.conv2(y)
        y = self.dropout2(self.activation(self.chomp2(y)))
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(y + res)


class TCNForecaster(nn.Module):
    """Dilated Temporal Convolutional Network for sequence forecasting."""

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        output_dim: int = 1,
        hidden_dim: int = 128,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        channel_sizes = [hidden_dim] * levels
        for i in range(levels):
            dilation = 2**i
            in_ch = input_dim if i == 0 else channel_sizes[i - 1]
            out_ch = channel_sizes[i]
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.head = nn.Linear(channel_sizes[-1], pred_len * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.transpose(1, 2)
        tcn_out = self.network(x_perm)
        last_timestep = tcn_out[:, :, -1]
        out = self.head(last_timestep)
        return out.view(x.size(0), self.pred_len, self.output_dim)


def build_tcn(
    input_dim: int,
    pred_len: int,
    output_dim: int,
    hidden_dim: int,
    levels: int,
    kernel_size: int,
    dropout: float,
) -> nn.Module:
    return TCNForecaster(
        input_dim=input_dim,
        pred_len=pred_len,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        levels=levels,
        kernel_size=kernel_size,
        dropout=dropout,
    )
