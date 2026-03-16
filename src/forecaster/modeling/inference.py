"""Inference utilities for the forecaster.

`get_forecast_data` is the **modular boundary** between the environment and
whatever market-data source is active.  To swap the data source, change only
this function — downstream logic (env, simulation) remains untouched.

Current backend: the physical load/price models in `src.demo.data_provider`.
Future backends could be an ML model, a live market API, or a CSV replay.
"""

from __future__ import annotations

from datetime import datetime

import joblib
import pandas as pd

from src.demo.data_provider import generate_hourly_timeline


def get_forecast_data(
    window_size: int = 48,
    start_date: datetime | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of market data with `price` and `demand` columns.

    This is the single entry-point the environment calls to fetch episode data.
    Swap the implementation here to change the data source without touching any
    downstream code.

    Args:
        window_size: Number of hourly steps needed (typically `max_steps`).
        start_date: Start of the window; defaults to now.

    Returns:
        DataFrame with at least `price` ($/kWh) and `demand` (kWh) columns,
        one row per hour, indexed 0..window_size-1.
    """
    if start_date is None:
        start_date = datetime.now()

    df = generate_hourly_timeline(start_date, num_hours=window_size)
    # Normalise to the two columns the env actually uses.
    return df[["price", "demand"]].reset_index(drop=True)


def load_model(path: str):
    """Load a serialised sklearn / compatible model from `path`."""
    return joblib.load(path)


def predict(model, df: pd.DataFrame):
    """Run inference on `df` using a trained forecasting model."""
    from src.forecaster.modeling.train_model import get_features_and_target

    X, _ = get_features_and_target(df)
    return model.predict(X)

