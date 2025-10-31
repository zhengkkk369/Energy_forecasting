from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class MissingValueHandler:
    strategy: str = "forward_fill"

    def __post_init__(self) -> None:
        self._strategies = {
            "forward_fill": self._forward_fill,
            "mean_impute": self._mean_impute,
        }

    def transform(self, features: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return imputed features and explicit missing mask."""
        handler = self._strategies.get(self.strategy, self._mean_impute)
        filled = handler(features.copy(), mask)
        return filled, mask.astype(np.float32)

    def _forward_fill(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        for i in range(1, data.shape[1]):
            missing = mask[:, i] == 0
            data[missing, i] = data[missing, i - 1]
        return data

    def _mean_impute(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mean = data.mean(axis=1, keepdims=True)
        missing = mask == 0
        data[missing] = np.broadcast_to(mean, data.shape)[missing]
        return data


def build_context(features: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Combine calendar and weather metadata into a single context dict."""
    context = {
        "time": meta.get("time_indices"),
        "calendar": meta.get("calendar_features"),
        "weather": meta.get("weather_features"),
        "missing_mask": meta.get("missing_mask"),
    }
    return {k: v for k, v in context.items() if v is not None}
