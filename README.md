# Energy Forecasting Concept Drift Framework

Skeleton codebase derived from the _energy_forecast_cl_drift_project.md_ specification.  
Implements placeholders for data streaming, adaptive modeling, drift detection, memory fusion, and online evaluation.

## D3A Integration (Detect -> Diagnose -> Adapt)

- **Two-stage drift pipeline**: `CandidateStage` raises high-sensitivity alarms which are confirmed by a longer-horizon KS test before adaptation is triggered.  
- **Drift-aware memory usage**: confirmed drifts write adapter snapshots to associative memory and retrieve layer-specific matches when adaptation is required.  
- **Adaptive scheduling hooks**: drift strength from D3A feeds the adaptation scheduler for gradual vs. abrupt responses.

Use `scripts/run_online_training.py` as the main entry point for experimentation.
