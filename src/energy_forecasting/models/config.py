from __future__ import annotations

from dataclasses import dataclass

@dataclass
class ModelConfig:
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
    rep_dim: int = 320
    seq_len: int = 96
    patch_len: int = 16
    freq_top_k: int = 16
    ts2vec_depth: int = 10
    ts2vec_gamma: float = 0.9

    def build(self):
        from .registry import create_model

        return create_model(self)


# Backwards compatibility alias
BaselineConfig = ModelConfig
