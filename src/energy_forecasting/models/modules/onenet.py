"""Wrapper that exposes the OneNet model through the module registry."""

from __future__ import annotations

from ..onenet.models import build_onenet

__all__ = ["build_onenet"]
