from __future__ import annotations

import numpy as np


class StandardScaler:
    """Numpy-based standard scaler compatible with the dataset loaders."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "StandardScaler":
        array = np.asarray(data, dtype=np.float32)
        self.mean = array.mean(axis=0)
        self.std = array.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        array = np.asarray(data, dtype=np.float32)
        return (array - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        array = np.asarray(data, dtype=np.float32)
        return array * self.std + self.mean

    def _check_fitted(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform().")
