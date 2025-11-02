from __future__ import annotations

from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmupScheduler:
    """Linear warmup before handing control to an optional scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        start_factor: float = 0.1,
        after_scheduler=None,
    ) -> None:
        if warmup_epochs < 1:
            raise ValueError("warmup_epochs must be >= 1")
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_factor = start_factor
        self.after_scheduler = after_scheduler
        self.current_epoch = 0
        self.base_lrs: List[float] = [
            group.get("initial_lr", group["lr"]) for group in optimizer.param_groups
        ]
        for lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = lr * self.start_factor
        self.finished = False

    def step(self, metric=None) -> None:
        if self.finished:
            if self.after_scheduler is None:
                return
            if isinstance(self.after_scheduler, ReduceLROnPlateau):
                self.after_scheduler.step(metric)
            else:
                self.after_scheduler.step()
            return

        self.current_epoch += 1
        progress = min(self.current_epoch / self.warmup_epochs, 1.0)
        factor = self.start_factor + progress * (1.0 - self.start_factor)
        for lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = lr * factor

        if self.current_epoch >= self.warmup_epochs:
            self.finished = True
            if self.after_scheduler is None:
                return
            # Reset state for downstream scheduler so it starts after warmup.
            if hasattr(self.after_scheduler, "base_lrs"):
                self.after_scheduler.base_lrs = self.base_lrs
            if hasattr(self.after_scheduler, "last_epoch"):
                self.after_scheduler.last_epoch = -1
