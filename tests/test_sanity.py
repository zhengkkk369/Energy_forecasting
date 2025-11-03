import pytest
import torch

from energy_forecasting.config import ProjectConfig
from energy_forecasting.models import BaselineConfig
from energy_forecasting.training.early_stopping import EarlyStopping
from energy_forecasting.training.ema import ModelEMA
from energy_forecasting.training.lr_utils import WarmupScheduler


def test_default_config() -> None:
    config = ProjectConfig()
    assert config.horizon == 24
    assert config.adapter.bottleneck_ratio == 0.25
    assert config.d3a.candidate_window == 48


def test_baseline_config_variants() -> None:
    model_types = (
        "lstm",
        "tcn",
        "transformer",
        "fsnet",
        "nomem",
        "ncca",
        "autoformer",
        "dlinear",
        "informer",
        "patchtst",
        "fedformer",
    )
    for model_type in model_types:
        cfg = BaselineConfig(
            input_dim=8,
            output_dim=1,
            pred_len=24,
            model_type=model_type,
            seq_len=96,
            patch_len=16,
            freq_top_k=8,
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


def test_warmup_scheduler_reaches_base_lr() -> None:
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = WarmupScheduler(optimizer, warmup_epochs=2, start_factor=0.5, after_scheduler=None)
    lrs = []
    for _ in range(3):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    assert pytest.approx(lrs[0], rel=1e-4) == 0.05
    assert pytest.approx(lrs[2], rel=1e-4) == 0.1


def test_model_ema_updates_shadow() -> None:
    model = torch.nn.Linear(3, 1)
    ema = ModelEMA(model, decay=0.5)
    original = {name: param.detach().clone() for name, param in model.named_parameters()}
    for param in model.parameters():
        param.data.add_(1.0)
    updated = {name: param.detach().clone() for name, param in model.named_parameters()}
    ema.update(model)
    for name, param in model.named_parameters():
        expected = original[name] * 0.5 + param.detach() * 0.5
        assert torch.allclose(ema.shadow[name], expected)
    ema.apply_shadow(model)
    for name, param in model.named_parameters():
        assert torch.allclose(param, ema.shadow[name])
    ema.restore(model)
    for name, param in model.named_parameters():
        assert torch.allclose(param, updated[name])
