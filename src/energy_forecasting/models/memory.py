from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Tuple

import torch
import torch.nn.functional as F

from .adapters import AdapterState


@dataclass
class MemoryItem:
    key: torch.Tensor
    state: AdapterState
    tag: str


class AdapterMemory:
    """Lightweight associative memory for adapter parameters."""

    def __init__(self, capacity: int = 2048, similarity: str = "cosine") -> None:
        self.capacity = capacity
        self.similarity = similarity
        self.storage: Deque[MemoryItem] = deque(maxlen=capacity)

    def write(self, key: torch.Tensor, state: AdapterState, tag: str = "default") -> None:
        self.storage.append(MemoryItem(key=key.detach().cpu(), state=state.clone(), tag=tag))

    def retrieve(self, key: torch.Tensor, top_k: int = 3, tag: str | None = None) -> Iterable[AdapterState]:
        if not self.storage:
            return []
        matches = self._similarities(key, tag)
        ranked = sorted(matches, key=lambda x: x[1], reverse=True)[:top_k]
        return [self.storage[idx].state.clone() for idx, _ in ranked]

    def _similarities(self, key: torch.Tensor, tag: str | None) -> Iterable[Tuple[int, float]]:
        q = key.detach().cpu().view(1, -1)
        scores = []
        for idx, item in enumerate(self.storage):
            if tag is not None and item.tag != tag:
                continue
            k = item.key.view(1, -1)
            if self.similarity == "cosine":
                score = F.cosine_similarity(q, k).item()
            else:
                score = torch.sum(-(q - k) ** 2).item()
            scores.append((idx, score))
        return scores
