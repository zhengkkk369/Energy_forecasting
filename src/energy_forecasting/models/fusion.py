from __future__ import annotations

from typing import Iterable

import torch

from .adapters import AdapterState


class AdapterFusion:
    """Combines current adapter parameters with retrieved memory states."""

    def __init__(self, alpha: float = 0.7) -> None:
        self.alpha = alpha

    def fuse(self, current: AdapterState, retrieved: Iterable[AdapterState]) -> AdapterState:
        aggregate = {name: tensor.clone() for name, tensor in current.params.items()}
        count = 0
        for state in retrieved:
            count += 1
            for name, tensor in state.params.items():
                aggregate[name] += tensor
        if count == 0:
            return current
        blended = {
            name: self.alpha * current.params[name] + (1 - self.alpha) * aggregate[name] / count
            for name in aggregate
        }
        return AdapterState(params={k: v.clone() for k, v in blended.items()})

    @staticmethod
    def snapshot(module: torch.nn.Module) -> AdapterState:
        return AdapterState({k: v.detach().clone() for k, v in module.state_dict().items()})

    @staticmethod
    def load(module: torch.nn.Module, state: AdapterState) -> None:
        module.load_state_dict(state.params, strict=False)
