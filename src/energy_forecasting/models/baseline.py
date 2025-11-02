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


class TS2VecForecaster(nn.Module):
    """Simplified TS2Vec-style encoder with convolutional front-end and GRU context."""

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
        # x: [batch, seq_len, input_dim]
        x_perm = x.transpose(1, 2)
        features = torch.relu(self.conv1(x_perm))
        features = self.norm(torch.relu(self.conv2(features)))
        features = features.transpose(1, 2)
        _, hidden = self.encoder(features)
        embedding = hidden[-1]
        out = self.head(embedding)
        return out.view(x.size(0), self.pred_len, self.output_dim)


class AutoformerForecaster(nn.Module):
    """Lightweight Autoformer-inspired model with series decomposition."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        output_dim: int = 1,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.moving_avg = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.encoder_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.seasonal_head = nn.Linear(d_model, pred_len * output_dim)
        self.trend_head = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        encoded = self.encoder(self.encoder_proj(seasonal))
        seasonal_out = self.seasonal_head(encoded[:, -1]).view(x.size(0), self.pred_len, self.output_dim)
        trend_component = self.trend_head(trend.mean(dim=-1))  # [B, pred_len]
        trend_out = trend_component.unsqueeze(-1).repeat(1, 1, self.output_dim)
        return seasonal_out + trend_out


class DLinearForecaster(nn.Module):
    """Implementation of the DLinear idea using seasonal/trend linear projections."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.seasonal_layer = nn.Linear(seq_len, pred_len)
        self.trend_layer = nn.Linear(seq_len, pred_len)
        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, seq_len, input_dim]
        x_t = x.transpose(1, 2)  # [B, input_dim, seq_len]
        seasonal = self.seasonal_layer(x_t)
        trend = self.trend_layer(x_t)
        combined = seasonal + trend  # [B, input_dim, pred_len]
        combined = combined.transpose(1, 2)  # [B, pred_len, input_dim]
        out = self.output_proj(combined)
        return out.view(x.size(0), self.pred_len, self.output_dim)


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
        encoded = self.encoder(self.positional_encoding(projected), mask=self._build_mask(x))
        out = self.head(encoded[:, -1])
        return out.view(x.size(0), self.pred_len, self.output_dim)

    def _build_mask(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        keep = max(1, seq_len // self.factor)
        mask = torch.zeros(seq_len, seq_len, device=x.device)
        if keep < seq_len:
            mask[:, : seq_len - keep] = float("-inf")
        return mask


class PatchTSTForecaster(nn.Module):
    """PatchTST-style patching followed by transformer encoder."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        output_dim: int = 1,
        patch_len: int = 16,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.patches = max(1, seq_len // patch_len)
        self.input_proj = nn.Linear(input_dim * patch_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, pred_len * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, feat = x.shape
        trimmed = x[:, -self.patches * self.patch_len :]
        patches = trimmed.view(batch, self.patches, self.patch_len * feat)
        tokens = self.input_proj(patches)
        encoded = self.encoder(tokens)
        out = self.head(encoded[:, -1])
        return out.view(batch, self.pred_len, self.output_dim)


class FEDformerForecaster(nn.Module):
    """Minimal FEDformer-style frequency-enhanced decoder."""

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        output_dim: int = 1,
        freq_top_k: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.freq_top_k = freq_top_k
        self.linear = nn.Linear(freq_top_k * input_dim * 2, hidden_dim)
        self.head = nn.Linear(hidden_dim, pred_len * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fourier transform along time dimension
        freq = torch.fft.rfft(x, dim=1)
        top_k = min(self.freq_top_k, freq.size(1))
        selected = freq[:, :top_k]
        if top_k < self.freq_top_k:
            pad_shape = (selected.size(0), self.freq_top_k - top_k, selected.size(2))
            pad = torch.zeros(pad_shape, dtype=selected.dtype, device=selected.device)
            selected = torch.cat([selected, pad], dim=1)
        features = torch.cat([selected.real, selected.imag], dim=-1)
        flattened = features.reshape(features.size(0), -1)
        rep = torch.relu(self.linear(flattened))
        out = self.head(rep)
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
    rep_dim: int = 128
    seq_len: int = 96
    patch_len: int = 16
    freq_top_k: int = 16

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
        if model_type == "ts2vec":
            return TS2VecForecaster(
                input_dim=self.input_dim,
                pred_len=self.pred_len,
                output_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                rep_dim=self.rep_dim,
            )
        if model_type == "autoformer":
            return AutoformerForecaster(
                input_dim=self.input_dim,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                output_dim=self.output_dim,
                d_model=self.d_model,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
            )
        if model_type == "dlinear":
            return DLinearForecaster(
                input_dim=self.input_dim,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                output_dim=self.output_dim,
            )
        if model_type == "informer":
            return InformerForecaster(
                input_dim=self.input_dim,
                pred_len=self.pred_len,
                output_dim=self.output_dim,
                d_model=self.d_model,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
            )
        if model_type == "patchtst":
            return PatchTSTForecaster(
                input_dim=self.input_dim,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                output_dim=self.output_dim,
                patch_len=self.patch_len,
                d_model=self.d_model,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
            )
        if model_type == "fedformer":
            return FEDformerForecaster(
                input_dim=self.input_dim,
                pred_len=self.pred_len,
                output_dim=self.output_dim,
                freq_top_k=self.freq_top_k,
                hidden_dim=self.hidden_dim,
            )
        raise ValueError(f"Unsupported model_type: {self.model_type}")
