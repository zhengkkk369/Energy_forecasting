from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from ..config import ProjectConfig
from ..data.datastream import DataStream, StreamBatch
from ..drift.adaptation import AdaptationScheduler
from ..drift.d3a import D3AController
from ..models.adapters import LayerAdapter
from ..models.film import FiLMConditioner
from ..models.fusion import AdapterFusion
from ..models.memory import AdapterMemory
from ..training.replay import ReplayBuffer


@dataclass
class OnlineTrainerState:
    step: int = 0
    last_drift_score: float = 0.0


class OnlineTrainer:
    """Coordinates data streaming, drift detection, and parameter updates."""

    def __init__(self, config: ProjectConfig, model: nn.Module, adapters: Iterable[LayerAdapter]) -> None:
        self.config = config
        self.model = model
        self.adapters = list(adapters)
        self.d3a = D3AController(config.d3a)
        self.scheduler = AdaptationScheduler()
        self.fusion = AdapterFusion(alpha=config.adapter.fusion_alpha)
        self.replay = ReplayBuffer(capacity=1024)
        self.memory = (
            AdapterMemory(capacity=config.memory.capacity, similarity=config.memory.similarity)
            if config.memory.enabled
            else None
        )
        self.state = OnlineTrainerState()
        self.fast_optimizer = torch.optim.AdamW(
            self._fast_params(), lr=config.fast_optimizer.lr, weight_decay=config.fast_optimizer.weight_decay
        )
        self.slow_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.slow_optimizer.lr, weight_decay=config.slow_optimizer.weight_decay
        )
        self.film = FiLMConditioner(d_in=16, d_model=128)

    def _fast_params(self) -> Iterable[torch.nn.Parameter]:
        params = []
        for adapter in self.adapters:
            params.extend(list(adapter.parameters()))
        return params

    def run(self, stream: DataStream, criterion: nn.Module) -> None:
        for batch in stream:
            self._predict(batch)
            if batch.target is None:
                continue
            loss = self._compute_loss(batch, criterion)
            self._update_fast(loss)
            self._update_slow(loss)
            self._update_memory(batch)
            self.state.step += 1

    def _predict(self, batch: StreamBatch) -> torch.Tensor:
        features = torch.as_tensor(batch.features, dtype=torch.float32)
        return self.model(features)

    def _compute_loss(self, batch: StreamBatch, criterion: nn.Module) -> torch.Tensor:
        predictions = self._predict(batch)
        target = torch.as_tensor(batch.target, dtype=torch.float32)
        return criterion(predictions, target)

    def _update_fast(self, loss: torch.Tensor) -> None:
        self.fast_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.fast_optimizer.step()

    def _update_slow(self, loss: torch.Tensor) -> None:
        self.slow_optimizer.zero_grad()
        loss.backward()
        self.slow_optimizer.step()

    def _update_memory(self, batch: StreamBatch) -> None:
        representation = torch.as_tensor(batch.features.mean(axis=1))
        signal, instruction = self.d3a.assess(representation.numpy())
        self.state.last_drift_score = signal.score
        _ = self.scheduler.step(instruction.drift_strength if instruction.trigger_adaptation else 0.0)

        if instruction.store_state and self.config.memory.enabled and self.memory is not None:
            key = self.d3a.build_key(batch.context)
            for idx, adapter in enumerate(self.adapters):
                snapshot = AdapterFusion.snapshot(adapter)
                self.memory.write(key, snapshot, tag=f"adapter_{idx}")

        if (
            instruction.trigger_adaptation
            and self.config.memory.enabled
            and instruction.use_memory
            and self.memory is not None
        ):
            key = self.d3a.build_key(batch.context)
            for idx, adapter in enumerate(self.adapters):
                retrieved = self.memory.retrieve(
                    key, top_k=self.config.memory.top_k, tag=f"adapter_{idx}"
                )
                current_state = AdapterFusion.snapshot(adapter)
                fused_state = self.fusion.fuse(current_state, retrieved)
                AdapterFusion.load(adapter, fused_state)

        self.replay.push(batch)
