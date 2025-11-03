from __future__ import annotations

from typing import Callable, Dict

from .config import ModelConfig
from .modules.autoformer import build_autoformer
from .modules.dlinear import build_dlinear
from .modules.fedformer import build_fedformer
from .modules.lstm import build_lstm
from .modules.patchtst import build_patchtst
from .modules.tcn import build_tcn
from .modules.transformers import build_informer, build_transformer
from .ts2vec.wrapper import build_fsnet_model, build_nomem_model, build_ncca_model


ModelBuilder = Callable[[ModelConfig], object]


def _build_lstm(cfg: ModelConfig):
    return build_lstm(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        pred_len=cfg.pred_len,
        output_dim=cfg.output_dim,
        dropout=cfg.dropout,
    )


def _build_tcn(cfg: ModelConfig):
    return build_tcn(
        input_dim=cfg.input_dim,
        pred_len=cfg.pred_len,
        output_dim=cfg.output_dim,
        hidden_dim=cfg.hidden_dim,
        levels=cfg.tcn_levels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    )


def _build_transformer(cfg: ModelConfig):
    return build_transformer(
        input_dim=cfg.input_dim,
        pred_len=cfg.pred_len,
        output_dim=cfg.output_dim,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
    )


def _build_fsnet(cfg: ModelConfig):
    return build_fsnet_model(
        input_dim=cfg.input_dim,
        pred_len=cfg.pred_len,
        output_dim=cfg.output_dim,
        rep_dim=cfg.rep_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.ts2vec_depth,
        gamma=cfg.ts2vec_gamma,
    )


def _build_nomem(cfg: ModelConfig):
    return build_nomem_model(
        input_dim=cfg.input_dim,
        pred_len=cfg.pred_len,
        output_dim=cfg.output_dim,
        rep_dim=cfg.rep_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.ts2vec_depth,
        gamma=cfg.ts2vec_gamma,
    )


def _build_ncca(cfg: ModelConfig):
    return build_ncca_model(
        input_dim=cfg.input_dim,
        pred_len=cfg.pred_len,
        output_dim=cfg.output_dim,
        rep_dim=cfg.rep_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.ts2vec_depth,
        gamma=cfg.ts2vec_gamma,
    )


def _build_autoformer(cfg: ModelConfig):
    return build_autoformer(
        input_dim=cfg.input_dim,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        output_dim=cfg.output_dim,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
    )


def _build_dlinear(cfg: ModelConfig):
    return build_dlinear(
        input_dim=cfg.input_dim,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        output_dim=cfg.output_dim,
    )


def _build_informer(cfg: ModelConfig):
    return build_informer(
        input_dim=cfg.input_dim,
        pred_len=cfg.pred_len,
        output_dim=cfg.output_dim,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    )


def _build_patchtst(cfg: ModelConfig):
    return build_patchtst(
        input_dim=cfg.input_dim,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        output_dim=cfg.output_dim,
        patch_len=cfg.patch_len,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
    )


def _build_fedformer(cfg: ModelConfig):
    return build_fedformer(
        input_dim=cfg.input_dim,
        pred_len=cfg.pred_len,
        output_dim=cfg.output_dim,
        freq_top_k=cfg.freq_top_k,
        hidden_dim=cfg.hidden_dim,
    )


MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    "lstm": _build_lstm,
    "tcn": _build_tcn,
    "transformer": _build_transformer,
    "fsnet": _build_fsnet,
    "nomem": _build_nomem,
    "ncca": _build_ncca,
    "autoformer": _build_autoformer,
    "dlinear": _build_dlinear,
    "informer": _build_informer,
    "patchtst": _build_patchtst,
    "fedformer": _build_fedformer,
}


MODEL_REGISTRY["ts2vec"] = _build_fsnet  # backward compatibility


def create_model(cfg: ModelConfig):
    model_type = cfg.model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")
    return MODEL_REGISTRY[model_type](cfg)
