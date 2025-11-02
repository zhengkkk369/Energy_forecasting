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


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


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
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.dropout1(self.relu1(self.chomp1(y)))
        y = self.conv2(y)
        y = self.dropout2(self.relu2(self.chomp2(y)))
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
        # x: [batch, seq_len, input_dim]
        x_perm = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        tcn_out = self.network(x_perm)
        last_timestep = tcn_out[:, :, -1]
        out = self.head(last_timestep)
        return out.view(x.size(0), self.pred_len, self.output_dim)


@dataclass
class BaselineConfig:
    input_dim: int
    output_dim: int
    pred_len: int
    model_type: str = "lstm"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    d_model: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    pooling: str = "last"
    tcn_levels: int = 4
    kernel_size: int = 3

    def build(self) -> nn.Module:
        model_type = self.model_type.lower()
        if model_type == "lstm":
            return LSTMForecaster(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                pred_len=self.pred_len,
                output_dim=self.output_dim,
                dropout=self.dropout,
            )
        if model_type == "transformer":
            return TransformerForecaster(
                input_dim=self.input_dim,
                pred_len=self.pred_len,
                output_dim=self.output_dim,
                d_model=self.d_model,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                pooling=self.pooling,
            )
        if model_type == "tcn":
            return TCNForecaster(
                input_dim=self.input_dim,
                pred_len=self.pred_len,
                output_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                levels=self.tcn_levels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            )
        raise ValueError(f"Unsupported model_type: {self.model_type}")
