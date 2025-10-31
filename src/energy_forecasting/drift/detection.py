from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict

import numpy as np
from scipy import stats


@dataclass
class DriftSignal:
    score: float
    triggered: bool
    metadata: Dict[str, float] = field(default_factory=dict)


class DriftDetector:
    """Tracks rolling statistics and raises alarms on distribution shifts."""

    def __init__(self, window: int = 168, quantile: float = 0.99) -> None:
        self.window = window
        self.quantile = quantile
        self.buffer: Deque[float] = deque(maxlen=window)

    def update(self, representation: np.ndarray) -> DriftSignal:
        distance = float(np.linalg.norm(representation))
        self.buffer.append(distance)
        threshold = self._threshold()
        triggered = distance > threshold
        return DriftSignal(score=distance, triggered=triggered, metadata={"threshold": threshold})

    def _threshold(self) -> float:
        if not self.buffer:
            return np.inf
        return float(np.quantile(list(self.buffer), self.quantile))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    return stats.ks_2samp(reference.flatten(), current.flatten()).statistic
