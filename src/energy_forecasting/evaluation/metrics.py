from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class MetricSnapshot:
    mae: float
    delta_spike: float
    recovery_time: int
    forgetting_rate: float


@dataclass
class OnlineMetrics:
    window: int = 24
    errors: List[float] = field(default_factory=list)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> MetricSnapshot:
        error = np.abs(prediction - target).mean()
        self.errors.append(error)
        delta_spike = self._spike(error)
        recovery = self._recovery_time()
        forgetting = self._forgetting_rate()
        return MetricSnapshot(
            mae=error,
            delta_spike=delta_spike,
            recovery_time=recovery,
            forgetting_rate=forgetting,
        )

    def summary(self) -> Dict[str, float]:
        if not self.errors:
            return {}
        arr = np.asarray(self.errors)
        return {"mae_mean": float(arr.mean()), "mae_std": float(arr.std())}

    def _spike(self, current: float) -> float:
        if len(self.errors) < 2:
            return 0.0
        baseline = np.median(self.errors[-self.window : -1])
        return current - baseline

    def _recovery_time(self) -> int:
        if not self.errors:
            return 0
        recent = self.errors[-self.window :]
        min_error = min(recent)
        return recent[::-1].index(min_error)

    def _forgetting_rate(self) -> float:
        if len(self.errors) < self.window:
            return 0.0
        return float(np.mean(self.errors[-self.window // 2 :]) - np.mean(self.errors[: self.window // 2]))
