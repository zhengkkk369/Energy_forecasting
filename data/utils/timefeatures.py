from __future__ import annotations

import numpy as np
import pandas as pd


def time_features(
    df_stamp: pd.DataFrame,
    timeenc: int = 0,
    freq: str = "h",
) -> np.ndarray:
    """Generate time-based covariates for forecasting datasets.

    Parameters
    ----------
    df_stamp:
        DataFrame with a `date` column containing timestamps.
    timeenc:
        0 returns integer calendar features; >0 returns sinusoidal encodings.
    freq:
        Sampling frequency string (unused for integer encoding but kept for compatibility).
    """

    if "date" not in df_stamp:
        raise ValueError("df_stamp must contain a 'date' column.")

    date = pd.to_datetime(df_stamp["date"])

    if timeenc == 0:
        features = np.stack(
            [
                date.dt.month.values,
                date.dt.day.values,
                date.dt.weekday.values,
                date.dt.hour.values if hasattr(date.dt, "hour") else np.zeros(len(date)),
            ],
            axis=1,
        )
        return features.astype(np.float32)

    # Sinusoidal encoding for smooth cyclical representation.
    month = date.dt.month.values
    day = date.dt.day.values
    weekday = date.dt.weekday.values
    hour = date.dt.hour.values if hasattr(date.dt, "hour") else np.zeros(len(date))

    features = np.stack(
        [
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            np.sin(2 * np.pi * day / 31),
            np.cos(2 * np.pi * day / 31),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
        ],
        axis=1,
    )
    return features.astype(np.float32)
