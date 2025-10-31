import numpy as np

from energy_forecasting.config import ProjectConfig
from energy_forecasting.drift.d3a import D3AController


def test_d3a_triggers_after_confirmation() -> None:
    config = ProjectConfig()
    controller = D3AController(config.d3a)
    triggered = False
    for step in range(config.d3a.confirmation_window + 5):
        value = 0.1 * step
        _, instruction = controller.assess(np.array([value], dtype=np.float32))
        if instruction.trigger_adaptation:
            triggered = True
            break
    assert triggered
