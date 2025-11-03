from __future__ import annotations

import torch
from torch import nn


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


def build_patchtst(
    input_dim: int,
    seq_len: int,
    pred_len: int,
    output_dim: int,
    patch_len: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    ff_dim: int,
) -> nn.Module:
    return PatchTSTForecaster(
        input_dim=input_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        output_dim=output_dim,
        patch_len=patch_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
    )
