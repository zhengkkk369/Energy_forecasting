from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .datastream import StreamBatch


@dataclass
class DriftConfig:
    drift_type: str  # "abrupt", "periodic", "gradual"
    start: int
    magnitude: float
    duration: Optional[int] = None  # None => persistent after start
    period: Optional[int] = None  # for periodic drift
    feature_indices: Optional[Sequence[int]] = None
    applies_to: str = "features"  # "features", "target", "both"
    mode: str = "additive"  # "additive" or "multiplicative"

    def __post_init__(self) -> None:
        if self.drift_type not in {"abrupt", "periodic", "gradual"}:
            raise ValueError(f"Unsupported drift_type: {self.drift_type}")
        if self.applies_to not in {"features", "target", "both"}:
            raise ValueError(f"Unsupported applies_to: {self.applies_to}")
        if self.mode not in {"additive", "multiplicative"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.drift_type == "periodic" and (self.period is None or self.period <= 0):
            raise ValueError("periodic drift requires a positive period.")
        if self.drift_type == "gradual" and (self.duration is None or self.duration <= 0):
            raise ValueError("gradual drift requires a positive duration.")


class DriftInjector:
    """Apply synthetic drift patterns (abrupt, periodic, gradual) to stream batches."""

    def __init__(self, configs: Iterable[DriftConfig]) -> None:
        self.configs: List[DriftConfig] = list(configs)

    def apply(self, step: int, batch: "StreamBatch") -> "StreamBatch":
        from .datastream import StreamBatch as _StreamBatch

        features = batch.features.copy()
        target = None if batch.target is None else batch.target.copy()

        for cfg in self.configs:
            if not self._is_active(cfg, step):
                continue
            delta = self._compute_delta(cfg, step)
            indices = self._resolve_indices(cfg, features.shape[-1])
            if cfg.applies_to in {"features", "both"}:
                features = self._apply_delta(features, delta, indices, cfg.mode)
            if cfg.applies_to in {"target", "both"} and target is not None:
                tgt_indices = self._resolve_indices(cfg, target.shape[-1])
                target = self._apply_delta(target, delta, tgt_indices, cfg.mode)

        return _StreamBatch(
            features=features,
            context=batch.context,
            target=target,
            timestamp=batch.timestamp,
        )

    def _is_active(self, cfg: DriftConfig, step: int) -> bool:
        if step < cfg.start:
            return False
        if cfg.duration is None or cfg.duration <= 0:
            return True
        return cfg.start <= step < cfg.start + cfg.duration

    def _compute_delta(self, cfg: DriftConfig, step: int) -> float:
        if cfg.drift_type == "abrupt":
            return cfg.magnitude
        if cfg.drift_type == "periodic":
            phase = (step - cfg.start) / cfg.period
            return cfg.magnitude * math.sin(2 * math.pi * phase)
        if cfg.drift_type == "gradual":
            progress = (step - cfg.start + 1) / cfg.duration
            progress = max(0.0, min(1.0, progress))
            return cfg.magnitude * progress
        raise RuntimeError("Unhandled drift type encountered.")

    def _resolve_indices(self, cfg: DriftConfig, feature_dim: int) -> np.ndarray:
        if cfg.feature_indices is None:
            return np.arange(feature_dim)
        return np.asarray(cfg.feature_indices, dtype=int)

    def _apply_delta(
        self,
        data: np.ndarray,
        delta: float,
        indices: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        result = data.copy()
        if mode == "additive":
            result[..., indices] += delta
        else:
            result[..., indices] *= (1.0 + delta)
        return result
