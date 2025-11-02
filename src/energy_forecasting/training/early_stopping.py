from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStoppingState:
    best_metric: float = float("inf")
    bad_epochs: int = 0


class EarlyStopping:
    """Stops training when validation metric stalls."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.min_delta = min_delta
        self.state = EarlyStoppingState()

    def step(self, metric: float) -> bool:
        """Return True when early stopping should trigger."""
        if metric < self.state.best_metric - self.min_delta:
            self.state.best_metric = metric
            self.state.bad_epochs = 0
            return False
        self.state.bad_epochs += 1
        return self.state.bad_epochs > self.patience

    def reset(self) -> None:
        self.state = EarlyStoppingState()
