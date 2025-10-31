from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptationState:
    decay_lambda: float
    soft_ratio: float


class AdaptationScheduler:
    """Adapts decay and soft update ratios based on detected drift strength."""

    def __init__(self, min_lambda: float = 1e-3, max_lambda: float = 5e-2) -> None:
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

    def step(self, drift_strength: float) -> AdaptationState:
        clipped = max(0.0, min(1.0, drift_strength))
        decay = self.min_lambda + clipped * (self.max_lambda - self.min_lambda)
        ratio = clipped
        return AdaptationState(decay_lambda=decay, soft_ratio=ratio)
