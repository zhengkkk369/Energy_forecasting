import numpy as np

from energy_forecasting.data.datastream import StreamBatch
from energy_forecasting.data.drift_injector import DriftConfig, DriftInjector


def make_batch(value: float = 0.0) -> StreamBatch:
    features = np.full((1, 12, 3), value, dtype=np.float32)
    target = np.full((1, 12, 1), value, dtype=np.float32)
    return StreamBatch(features=features, context={}, target=target, timestamp=0)


def test_abrupt_drift_additive() -> None:
    cfg = DriftConfig(drift_type="abrupt", start=5, magnitude=2.0, applies_to="features")
    injector = DriftInjector([cfg])

    batch = make_batch()
    early = injector.apply(4, batch)
    assert np.allclose(early.features, batch.features)

    drifted = injector.apply(5, batch)
    assert np.allclose(drifted.features, batch.features + 2.0)


def test_periodic_multiplicative() -> None:
    cfg = DriftConfig(
        drift_type="periodic",
        start=0,
        magnitude=0.5,
        period=4,
        applies_to="both",
        mode="multiplicative",
    )
    injector = DriftInjector([cfg])
    batch = make_batch(1.0)

    step0 = injector.apply(0, batch)
    assert np.allclose(step0.features, batch.features)  # sin(0) = 0

    step1 = injector.apply(1, batch)
    expected = batch.features * (1 + 0.5 * np.sin(2 * np.pi * (1 / 4)))
    assert np.allclose(step1.features, expected)
    assert np.allclose(step1.target, expected[..., :1])


def test_gradual_drift_progression() -> None:
    cfg = DriftConfig(drift_type="gradual", start=2, duration=4, magnitude=1.0, feature_indices=[0])
    injector = DriftInjector([cfg])
    batch = make_batch()

    no_drift = injector.apply(1, batch)
    assert np.allclose(no_drift.features, batch.features)

    mid = injector.apply(4, batch)
    progress = ((4 - 2 + 1) / 4) * 1.0
    assert np.allclose(mid.features[..., 0], batch.features[..., 0] + progress)
