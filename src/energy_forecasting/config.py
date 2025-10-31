from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class DriftDetectionConfig:
    window: int = 168
    threshold_quantile: float = 0.99
    detector: str = "wasserstein"


@dataclass
class MemoryConfig:
    enabled: bool = True
    capacity: int = 2048
    top_k: int = 3
    similarity: str = "cosine"


@dataclass
class AdapterConfig:
    bottleneck_ratio: float = 0.25
    fusion_alpha: float = 0.7


@dataclass
class D3AConfig:
    candidate_window: int = 48
    confirmation_window: int = 168
    sensitivity_z: float = 2.0
    confirmation_threshold: float = 0.3
    cooldown: int = 12


@dataclass
class ProjectConfig:
    seed: int = 1024
    horizon: int = 24
    backbone: str = "temporal_transformer"
    fast_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1e-3))
    slow_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=3e-5))
    drift: DriftDetectionConfig = field(default_factory=DriftDetectionConfig)
    d3a: D3AConfig = field(default_factory=D3AConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    features: List[str] = field(default_factory=lambda: ["load", "wind", "solar", "temperature"])
    context_features: List[str] = field(
        default_factory=lambda: ["hour", "day_of_week", "holiday", "weather_mask"]
    )
