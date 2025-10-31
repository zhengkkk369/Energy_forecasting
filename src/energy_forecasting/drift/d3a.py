from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import numpy as np
import torch

from ..config import D3AConfig
from .detection import DriftDetector, DriftSignal, ks_statistic


@dataclass
class D3AInstruction:
    trigger_adaptation: bool
    drift_strength: float
    drift_type: str
    use_memory: bool
    store_state: bool
    diagnostics: Dict[str, float]


class CandidateStage:
    """High-sensitivity detector that proposes candidate drifts."""

    def __init__(self, window: int, sensitivity_z: float) -> None:
        self.window = window
        self.sensitivity_z = sensitivity_z
        self.distances: Deque[float] = deque(maxlen=window)

    def update(self, value: float) -> Tuple[bool, Dict[str, float]]:
        self.distances.append(value)
        if len(self.distances) < self.window:
            return False, {"z_score": 0.0}
        mean = float(np.mean(self.distances))
        std = float(np.std(self.distances) + 1e-6)
        z_score = (value - mean) / std
        triggered = z_score >= self.sensitivity_z
        return triggered, {"z_score": z_score, "mean": mean, "std": std}


class ConfirmationStage:
    """Confirms drift using distributional discrepancy over a longer horizon."""

    def __init__(self, window: int, threshold: float) -> None:
        self.window = window
        self.threshold = threshold
        self.buffer: Deque[float] = deque(maxlen=window)

    def update(self, value: float) -> None:
        self.buffer.append(value)

    def confirm(self) -> Tuple[bool, Dict[str, float]]:
        if len(self.buffer) < self.window:
            return False, {"ks_stat": 0.0}
        half = self.window // 2
        pre = np.array(list(self.buffer)[:half])
        post = np.array(list(self.buffer)[half:])
        ks_stat = ks_statistic(pre, post)
        return ks_stat >= self.threshold, {"ks_stat": ks_stat}

    def classify(self) -> str:
        if len(self.buffer) < self.window:
            return "unknown"
        half = self.window // 2
        pre = np.array(list(self.buffer)[:half])
        post = np.array(list(self.buffer)[half:])
        trend = np.mean(post) - np.mean(pre)
        return "abrupt" if abs(trend) > 0.5 else "gradual"


class D3AController:
    """Implements detect-before-adapt (D3A) logic around a base detector."""

    def __init__(self, config: D3AConfig, detector: DriftDetector | None = None) -> None:
        self.config = config
        self.detector = detector or DriftDetector(window=config.confirmation_window, quantile=0.99)
        self.candidate = CandidateStage(config.candidate_window, config.sensitivity_z)
        self.confirmation = ConfirmationStage(config.confirmation_window, config.confirmation_threshold)
        self.cooldown = 0

    def assess(self, representation: np.ndarray) -> Tuple[DriftSignal, D3AInstruction]:
        base_signal = self.detector.update(representation)
        self.confirmation.update(base_signal.score)
        candidate_flag, candidate_meta = self.candidate.update(base_signal.score)

        trigger = False
        drift_type = "none"
        diagnostics = {"distance": base_signal.score, **candidate_meta}

        if self.cooldown > 0:
            self.cooldown -= 1
        elif candidate_flag:
            confirmed, confirm_meta = self.confirmation.confirm()
            diagnostics.update(confirm_meta)
            if confirmed:
                trigger = True
                drift_type = self.confirmation.classify()
                self.cooldown = self.config.cooldown
        else:
            diagnostics.update({"ks_stat": diagnostics.get("ks_stat", 0.0)})

        strength = float(np.tanh(base_signal.score))
        instruction = D3AInstruction(
            trigger_adaptation=trigger,
            drift_strength=strength,
            drift_type=drift_type,
            use_memory=trigger,
            store_state=trigger or candidate_flag,
            diagnostics=diagnostics,
        )
        return base_signal, instruction

    @staticmethod
    def build_key(context: Dict[str, np.ndarray | torch.Tensor]) -> torch.Tensor:
        if not context:
            return torch.zeros(1)
        vectors = []
        for value in context.values():
            tensor = torch.as_tensor(value, dtype=torch.float32).view(1, -1)
            vectors.append(tensor.mean(dim=-1, keepdim=True))
        stacked = torch.cat(vectors, dim=-1)
        return stacked.mean(dim=0)
