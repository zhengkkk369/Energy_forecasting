"""Experiment helpers for the OneNet model.

The original OneNet codebase uses a separate experiment abstraction to glue
configuration parsing and model creation.  Within this project the builder is
used instead, but we keep a lightweight experiment wrapper to ease
integration and to aid reproducibility in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .models import OneNet, OneNetConfig


@dataclass
class OneNetExperiment:
    """Container holding the configuration and instantiated model."""

    config: OneNetConfig
    model: OneNet

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "OneNetExperiment":
        """Build an experiment from a generic configuration dictionary."""

        config = OneNetConfig(**cfg_dict)
        model = OneNet(config)
        return cls(config=config, model=model)

    def state_dict(self) -> Dict[str, Any]:
        """Return a serialisable experiment state."""

        return {
            "config": self.config.__dict__,
            "model_state": self.model.state_dict(),
        }
