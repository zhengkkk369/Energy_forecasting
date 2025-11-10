"""Shared helpers for command-line entry points in ``scripts/``.

This module centralises logic that was previously duplicated across the
individual command line scripts.  Keeping the dataset registry, default
hyper-parameters and argument-parsing utilities in one place reduces the
chance of the files drifting out of sync and simplifies future updates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Mapping, Tuple

# Ensure the repository and ``src`` directory are importable when a script is
# executed directly via ``python scripts/xyz.py``.
ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
for path in (ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from data.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute  # noqa: E402

# Mapping of dataset identifiers to their loader class and the static metadata
# required to instantiate the dataset.  Several scripts rely on the same
# configuration, so it lives here.
DATASET_REGISTRY: Mapping[str, Tuple[type, Mapping[str, str]]] = {
    "ETTh1": (Dataset_ETT_hour, {"data_path": "ETTh1.csv", "freq": "h", "target": "OT"}),
    "ETTh2": (Dataset_ETT_hour, {"data_path": "ETTh2.csv", "freq": "h", "target": "OT"}),
    "ETTm1": (Dataset_ETT_minute, {"data_path": "ETTm1.csv", "freq": "15min", "target": "OT"}),
    "ETTm2": (Dataset_ETT_minute, {"data_path": "ETTm2.csv", "freq": "15min", "target": "OT"}),
    "ECL": (Dataset_Custom, {"data_path": "ECL.csv", "freq": "h", "target": "MT_320"}),
    "WTH": (Dataset_Custom, {"data_path": "WTH.csv", "freq": "h", "target": "WetBulbCelsius"}),
}

# Dataset-specific defaults override command line defaults when the user has
# not provided explicit values.  Only keys shared by the command line scripts
# are included here.
DATASET_DEFAULTS: Mapping[str, Mapping[str, float | int]] = {
    "ETTh1": {"seq_len": 336, "label_len": 168, "pred_len": 96, "learning_rate": 1e-3, "d_model": 256},
    "ETTh2": {"seq_len": 336, "label_len": 168, "pred_len": 96, "learning_rate": 1e-3, "d_model": 256},
}


def parse_kernel_sizes(value: str) -> Tuple[int, ...]:
    """Parse a comma-separated list of integers used for convolution kernels."""

    try:
        items = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise argparse.ArgumentTypeError("Kernel sizes must be comma-separated integers") from exc

    if not items:
        raise argparse.ArgumentTypeError("At least one kernel size must be provided")

    return tuple(items)


def apply_dataset_defaults(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    dataset_defaults: Mapping[str, Mapping[str, float | int]] | None = None,
    required_keys: Iterable[str] | None = None,
) -> argparse.Namespace:
    """Update ``args`` with dataset-specific defaults when appropriate.

    Parameters
    ----------
    parser:
        The parser used to create ``args``.  ``parser.get_default`` is required
        to determine whether a value was explicitly supplied by the user.
    args:
        The parsed arguments instance to mutate.
    dataset_defaults:
        Optional mapping overriding :data:`DATASET_DEFAULTS`.
    required_keys:
        Optional iterable restricting which keys from the dataset defaults are
        applied.  This is primarily useful for scripts whose argument names
        deviate slightly.
    """

    defaults = DATASET_DEFAULTS if dataset_defaults is None else dataset_defaults
    dataset = defaults.get(args.dataset)
    if not dataset:
        return args

    for key, value in dataset.items():
        if required_keys is not None and key not in required_keys:
            continue
        if not hasattr(args, key):
            continue
        default_value = parser.get_default(key)
        if getattr(args, key) == default_value:
            setattr(args, key, value)
    return args


__all__ = [
    "DATASET_DEFAULTS",
    "DATASET_REGISTRY",
    "apply_dataset_defaults",
    "parse_kernel_sizes",
]

