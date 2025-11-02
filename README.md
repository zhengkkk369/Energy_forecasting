# Energy Forecasting Concept Drift Framework

Skeleton codebase derived from the _energy_forecast_cl_drift_project.md_ specification.  
Implements placeholders for data streaming, adaptive modeling, drift detection, memory fusion, and online evaluation.

## D3A Integration (Detect -> Diagnose -> Adapt)

- **Two-stage drift pipeline**: `CandidateStage` raises high-sensitivity alarms which are confirmed by a longer-horizon KS test before adaptation is triggered.  
- **Drift-aware memory usage**: confirmed drifts write adapter snapshots to associative memory and retrieve layer-specific matches when adaptation is required.  
- **Adaptive scheduling hooks**: drift strength from D3A feeds the adaptation scheduler for gradual vs. abrupt responses.

Use `scripts/run_online_training.py` as the main entry point for experimentation.

## Running the Baseline Experiment

- Place the raw CSV datasets under the `data/` directory (already includes ETT/ECL/WTH samples).
- Execute a baseline run (CPU example):
  ```
  python scripts/run_online_training.py --dataset ETTh1 --model lstm --epochs 1 --batch-size 16 --device cpu
  ```
- Switch model families with `--model` (`lstm`, `tcn`, `transformer`) and adjust hyperparameters such as `--hidden-dim`, `--tcn-levels`, or `--num-heads` to explore architecture variants.
- Optional training utilities: enable schedulers (`--scheduler step|cosine|plateau`) and early stopping (`--early-stopping-patience`, `--early-stopping-min-delta`) for longer runs.
- Accuracy helpers: leverage warmup (`--warmup-epochs`, `--warmup-start-factor`), gradient clipping (`--grad-clip-norm`), and EMA averaging (`--ema-decay`) to stabilise training.
- Logs report training/validation curves and final test MAE for the configured forecaster.

## Synthetic Drift Injection

Use `energy_forecasting.data.drift_injector.DriftInjector` to inject abrupt, periodic, or gradual shifts into the online stream:

```python
from energy_forecasting.data import DataStream, DriftInjector, DriftConfig

injector = DriftInjector([
    DriftConfig(drift_type="abrupt", start=200, magnitude=1.5),
    DriftConfig(drift_type="periodic", start=0, magnitude=0.3, period=168, mode="multiplicative"),
    DriftConfig(drift_type="gradual", start=500, duration=240, magnitude=-0.8, feature_indices=[0, 1]),
])
stream = DataStream(horizon=96, lag=4, drift_injector=injector)
```
