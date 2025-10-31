from energy_forecasting.config import ProjectConfig


def test_default_config() -> None:
    config = ProjectConfig()
    assert config.horizon == 24
    assert config.adapter.bottleneck_ratio == 0.25
    assert config.d3a.candidate_window == 48
