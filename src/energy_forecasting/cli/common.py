"""Shared command-line helpers for Energy Forecasting scripts."""

from __future__ import annotations

import argparse
from typing import Mapping, Tuple

from data.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute

MODEL_CHOICES = [
    "lstm",
    "tcn",
    "transformer",
    "fsnet",
    "nomem",
    "ncca",
    "autoformer",
    "dlinear",
    "informer",
    "patchtst",
    "fedformer",
    "onenet",
]

DATASET_REGISTRY = {
    "ETTh1": (Dataset_ETT_hour, {"data_path": "ETTh1.csv", "freq": "h", "target": "OT"}),
    "ETTh2": (Dataset_ETT_hour, {"data_path": "ETTh2.csv", "freq": "h", "target": "OT"}),
    "ETTm1": (Dataset_ETT_minute, {"data_path": "ETTm1.csv", "freq": "15min", "target": "OT"}),
    "ETTm2": (Dataset_ETT_minute, {"data_path": "ETTm2.csv", "freq": "15min", "target": "OT"}),
    "ECL": (Dataset_Custom, {"data_path": "ECL.csv", "freq": "h", "target": "MT_320"}),
    "WTH": (Dataset_Custom, {"data_path": "WTH.csv", "freq": "h", "target": "WetBulbCelsius"}),
}

DATASET_DEFAULTS = {
    "ETTh1": {
        "seq_len": 336,
        "label_len": 168,
        "pred_len": 96,
        "learning_rate": 1e-3,
        "d_model": 256,
    },
    "ETTh2": {
        "seq_len": 336,
        "label_len": 168,
        "pred_len": 96,
        "learning_rate": 1e-3,
        "d_model": 256,
    },
}


def parse_kernel_sizes(value: str) -> Tuple[int, ...]:
    """Parse a comma separated list of kernel sizes."""

    try:
        items = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:  # pragma: no cover - argparse handles surfacing
        raise argparse.ArgumentTypeError("Kernel sizes must be comma-separated integers") from exc
    if not items:
        raise argparse.ArgumentTypeError("At least one kernel size must be provided")
    return tuple(items)


def apply_dataset_overrides(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    dataset_defaults: Mapping[str, Mapping[str, float]] | None = None,
) -> None:
    """Update argument values when users kept dataset dependent defaults.

    The helper mirrors the logic used by the scripts historically: only override an
    argument when the caller did not explicitly change it from the parser default.
    """

    dataset_defaults = DATASET_DEFAULTS if dataset_defaults is None else dataset_defaults
    if not hasattr(args, "dataset"):
        return
    overrides = dataset_defaults.get(args.dataset)
    if overrides is None:
        return
    for key, value in overrides.items():
        if not hasattr(args, key):
            continue
        default_value = parser.get_default(key)
        if getattr(args, key) == default_value:
            setattr(args, key, value)
