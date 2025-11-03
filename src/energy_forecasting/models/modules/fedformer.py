from __future__ import annotations

import torch
from torch import nn


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


def build_fedformer(
    input_dim: int,
    pred_len: int,
    output_dim: int,
    freq_top_k: int,
    hidden_dim: int,
) -> nn.Module:
    return FEDformerForecaster(
        input_dim=input_dim,
        pred_len=pred_len,
        output_dim=output_dim,
        freq_top_k=freq_top_k,
        hidden_dim=hidden_dim,
    )
