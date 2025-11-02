from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

@dataclass
class StreamBatch:
    features: np.ndarray
    context: Dict[str, np.ndarray]
    target: Optional[np.ndarray] = None
    timestamp: Optional[int] = None


class DataStream:
    """Simulates online data access with optional label delays and drift injection."""

    def __init__(
        self,
        horizon: int,
        lag: int = 0,
        drift_injector: Optional[object] = None,
    ) -> None:
        self.horizon = horizon
        self.lag = lag
        self._cursor = 0
        self.drift_injector = drift_injector

    def __iter__(self) -> Iterator[StreamBatch]:
        return self

    def __next__(self) -> StreamBatch:
        if self._cursor >= self._max_steps():
            raise StopIteration
        batch = self.get_batch(self._cursor)
        self._cursor += 1
        return batch

    def reset(self) -> None:
        self._cursor = 0

    def get_batch(self, step: int) -> StreamBatch:
        """Return synthetic placeholder batch; replace with real stream binding."""
        features = np.random.randn(1, self.horizon, 8)
        context = {
            "time": np.random.randint(0, 24, size=(1, self.horizon)),
            "weather": np.random.randn(1, self.horizon, 4),
            "mask": np.random.choice([0, 1], size=(1, self.horizon, 1)),
        }
        target = None if step < self.lag else np.random.randn(1, self.horizon, 1)
        batch = StreamBatch(features=features, context=context, target=target, timestamp=step)
        if self.drift_injector is not None:
            batch = self.drift_injector.apply(step, batch)
        return batch

    def _max_steps(self) -> int:
        """Placeholder stream length."""
        return 1000
