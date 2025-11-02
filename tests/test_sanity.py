import torch

from energy_forecasting.config import ProjectConfig
from energy_forecasting.models.baseline import BaselineConfig


def test_default_config() -> None:
    config = ProjectConfig()
    assert config.horizon == 24
    assert config.adapter.bottleneck_ratio == 0.25
    assert config.d3a.candidate_window == 48


def test_baseline_config_variants() -> None:
    for model_type in ("lstm", "tcn", "transformer"):
        cfg = BaselineConfig(
            input_dim=8,
            output_dim=1,
            pred_len=24,
            model_type=model_type,
        )
        model = cfg.build()
        dummy_input = torch.randn(2, 96, 8)
        out = model(dummy_input)
        assert out.shape == (2, 24, 1)
