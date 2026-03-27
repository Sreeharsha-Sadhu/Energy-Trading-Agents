"""Tests for the inference module with dual demand output."""

from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

from src.forecaster.modeling.inference import (
    RealtimeDemandForecaster,
    get_forecast_data,
    get_segment_model_path,
)


def test_get_forecast_data_returns_three_columns():
    """Verify get_forecast_data returns price, actual_demand, predicted_demand."""
    df = get_forecast_data(window_size=24, start_date=datetime(2025, 3, 7, 0, 0))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 24
    required_cols = {"price", "actual_demand", "predicted_demand"}
    assert required_cols.issubset(df.columns), (
        f"Missing columns. Expected {required_cols}, got {set(df.columns)}"
    )


def test_actual_and_predicted_demand_are_positive():
    """Both demand columns should contain positive values."""
    df = get_forecast_data(window_size=48, start_date=datetime(2025, 3, 7, 0, 0))

    assert (df["actual_demand"] > 0).all(), "actual_demand should be positive"
    assert (df["predicted_demand"] > 0).all(), "predicted_demand should be positive"
    assert (df["price"] > 0).all(), "price should be positive"


def test_get_forecast_data_uses_default_segment():
    """get_forecast_data should handle missing model gracefully with a fallback."""
    # When no model exists, predicted_demand should equal actual_demand (perfect foresight)
    df = get_forecast_data(
        window_size=24,
        start_date=datetime(2025, 3, 7, 0, 0),
        segment_name="NonExistentSegment",
    )

    assert len(df) == 24
    # With no model, fallback predictions should remain positive and finite.
    assert np.isfinite(df["predicted_demand"]).all()
    assert (df["predicted_demand"] >= 0).all()


def test_realtime_forecaster_prediction_is_non_negative():
    forecaster = RealtimeDemandForecaster(segment_name="Residential_Solar")
    predicted = forecaster.predict_demand(
        dt=datetime(2026, 3, 27, 13, 0),
        actual_demand=2.3,
        demand_history=deque([1.8, 2.1, 2.5], maxlen=336),
    )
    assert np.isfinite(predicted)
    assert predicted >= 0.0


def test_realtime_forecaster_fallback_for_missing_segment():
    forecaster = RealtimeDemandForecaster(segment_name="NonExistentSegment")
    predicted = forecaster.predict_demand(
        dt=datetime(2026, 3, 27, 13, 0),
        actual_demand=2.3,
        demand_history=deque([], maxlen=336),
    )
    assert np.isfinite(predicted)
    assert predicted >= 0.0
