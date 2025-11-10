"""CLI helpers shared across training scripts."""

from .common import (
    DATASET_DEFAULTS,
    DATASET_REGISTRY,
    MODEL_CHOICES,
    apply_dataset_overrides,
    parse_kernel_sizes,
)

__all__ = [
    "DATASET_DEFAULTS",
    "DATASET_REGISTRY",
    "MODEL_CHOICES",
    "apply_dataset_overrides",
    "parse_kernel_sizes",
]
