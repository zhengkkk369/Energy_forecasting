import torch

from energy_forecasting.config import ProjectConfig
from energy_forecasting.models.baseline import BaselineConfig
from energy_forecasting.training.early_stopping import EarlyStopping


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


def test_early_stopping_triggers() -> None:
    stopper = EarlyStopping(patience=2, min_delta=0.05)
    metrics = [1.0, 0.96, 0.955, 0.954, 0.953]
    triggered = False
    for metric in metrics:
        if stopper.step(metric):
            triggered = True
            break
    assert triggered
