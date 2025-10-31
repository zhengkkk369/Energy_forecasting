"""Core package for energy forecasting with concept drift handling."""

from .config import ProjectConfig  # noqa: F401
from .training.online_loop import OnlineTrainer  # noqa: F401

__all__ = ["ProjectConfig", "OnlineTrainer"]
