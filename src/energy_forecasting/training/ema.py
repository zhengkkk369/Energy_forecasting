from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterator, Tuple

import torch
from torch import nn


class ModelEMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be between 0 and 1.")
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._init_shadow(model)

    def _init_shadow(self, model: nn.Module) -> None:
        for name, param in self._named_parameters(model):
            self.shadow[name] = param.detach().clone()

    def _named_parameters(self, model: nn.Module) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, param in model.named_parameters():
            if param.requires_grad:
                yield name, param

    def update(self, model: nn.Module) -> None:
        for name, param in self._named_parameters(model):
            assert name in self.shadow, f"Parameter {name} missing in EMA shadow."
            shadow_param = self.shadow[name]
            shadow_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in self._named_parameters(model):
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module) -> None:
        for name, param in self._named_parameters(model):
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return OrderedDict((name, tensor.detach().clone()) for name, tensor in self.shadow.items())

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.shadow = OrderedDict((name, tensor.detach().clone()) for name, tensor in state_dict.items())

    def copy_to_model(self, model: nn.Module) -> None:
        for name, param in self._named_parameters(model):
            if name in self.shadow:
                param.data.copy_(self.shadow[name].data)
