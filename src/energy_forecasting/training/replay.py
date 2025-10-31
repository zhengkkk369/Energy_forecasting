from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List, Optional

from ..data.datastream import StreamBatch


class ReplayBuffer:
    """Holds recent batches for rehearsal or correction."""

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self.storage: Deque[StreamBatch] = deque(maxlen=capacity)

    def push(self, batch: StreamBatch) -> None:
        self.storage.append(batch)

    def sample(self, k: int = 4) -> List[StreamBatch]:
        if not self.storage:
            return []
        k = min(k, len(self.storage))
        return list(list(self.storage)[-k:])

    def __len__(self) -> int:
        return len(self.storage)
