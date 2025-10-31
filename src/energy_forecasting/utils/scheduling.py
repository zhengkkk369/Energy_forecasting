from __future__ import annotations


def exponential_decay(step: int, base: float = 0.01, rate: float = 0.001) -> float:
    return base * (1.0 - rate) ** step
