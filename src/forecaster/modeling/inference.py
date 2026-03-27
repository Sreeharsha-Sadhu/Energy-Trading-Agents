"""Inference utilities for the forecaster.

`get_forecast_data` is the **modular boundary** between the environment and
whatever market-data source is active.  To swap the data source, change only
this function — downstream logic (env, simulation) remains untouched.

Current backend: the physical load/price models in `src.demo.data_provider`,
enriched with predicted demand from trained ML models.
Future backends could be a live market API, a CSV replay, or another forecaster.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.demo.data_provider import generate_hourly_timeline
from src.forecaster.data_generation.solar_model import get_solar_irradiance

logger = logging.getLogger(__name__)

# Default segment to use when loading the forecasting model.
# Can be overridden by segment_name parameter.
DEFAULT_SEGMENT = "Residential_Solar"
_DEFAULT_FEATURE_VALUES = {
    "Id": 0.0,
    "MeterCount": 200.0,
    "LoadMeterCount": 150.0,
    "GenMeterCount": 50.0,
    "Hour": 0.0,
    "DayOfWeek": 0.0,
    "Month": 1.0,
    "Quarter": 1.0,
    "Year": 2026.0,
    "DayOfYear": 1.0,
    "WeekOfYear": 1.0,
    "IsWeekend": 0.0,
    "Temperature": 65.0,
    "Solar_Irradiance": 0.0,
    "BaseLoad_lag_168h": 1.2,
    "BaseLoad_rolling_mean_7d": 1.2,
}


def _estimate_temperature(dt: datetime) -> float:
    """Estimate temperature deterministically from seasonality and hour."""
    base_temp_f = 65.0
    seasonal_effect = -15.0 * math.cos(
        2.0 * math.pi * (dt.timetuple().tm_yday - 30) / 365.25
    )
    diurnal_effect = -5.0 * math.cos(2.0 * math.pi * dt.hour / 24.0)
    return float(base_temp_f + seasonal_effect + diurnal_effect)


def _load_segment_feature_defaults(segment_name: str) -> dict[str, float]:
    """Load median feature defaults from a segment CSV when available."""
    defaults = dict(_DEFAULT_FEATURE_VALUES)
    segment_csv = Path("data") / "segments" / f"{segment_name}.csv"
    if not segment_csv.exists():
        return defaults

    try:
        df = pd.read_csv(segment_csv)
    except Exception:
        return defaults

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        defaults[column] = float(df[column].median())
    return defaults


class RealtimeDemandForecaster:
    """Forecast demand from realtime ticks using a trained segment model."""

    def __init__(self, segment_name: str | None = None):
        self.segment_name = segment_name or settings.DEMO_FORECAST_SEGMENT
        self.model = None
        self.feature_names: list[str] = []
        self.feature_defaults = _load_segment_feature_defaults(self.segment_name)
        base_load_median = float(self.feature_defaults.get("BaseLoad", 1.0))
        # Convert model-scale loads (BaseLoad target scale) to demo-scale kWh.
        self.model_to_demo_scale = max(base_load_median / 2.0, 1.0)
        self._load_model()

    def _load_model(self) -> None:
        """Load champion model and cache feature names for online prediction."""
        try:
            model_path = get_segment_model_path(self.segment_name)
            self.model = load_model(str(model_path))
            raw_names = list(getattr(self.model, "feature_names_in_", []))
            self.feature_names = [str(name) for name in raw_names]
            logger.info(
                "Realtime forecaster loaded model %s for segment %s",
                model_path,
                self.segment_name,
            )
        except Exception as exc:
            logger.warning(
                "Realtime forecaster unavailable for segment %s: %s",
                self.segment_name,
                exc,
            )
            self.model = None
            self.feature_names = []

    def _build_feature_row(
        self,
        dt: datetime,
        actual_demand: float,
        demand_history: deque[float],
    ) -> pd.DataFrame:
        """Build a one-row DataFrame matching model training feature schema."""
        lag_window = settings.DEMO_FORECAST_HISTORY_HOURS
        lag_demand = actual_demand
        if len(demand_history) >= lag_window:
            lag_demand = float(demand_history[-lag_window])

        if demand_history:
            rolling_values = list(demand_history)[-lag_window:]
            rolling_mean = float(np.mean(rolling_values))
        else:
            rolling_mean = actual_demand

        feature_row = dict(self.feature_defaults)
        feature_row.update(
            {
                "Hour": float(dt.hour),
                "DayOfWeek": float(dt.weekday()),
                "Month": float(dt.month),
                "Quarter": float((dt.month - 1) // 3 + 1),
                "Year": float(dt.year),
                "DayOfYear": float(dt.timetuple().tm_yday),
                "WeekOfYear": float(dt.isocalendar().week),
                "IsWeekend": float(1 if dt.weekday() >= 5 else 0),
                "Temperature": _estimate_temperature(dt),
                "Solar_Irradiance": float(get_solar_irradiance(dt)),
                "BaseLoad_lag_168h": float(lag_demand * self.model_to_demo_scale),
                "BaseLoad_rolling_mean_7d": float(
                    rolling_mean * self.model_to_demo_scale
                ),
            }
        )

        columns = (
            self.feature_names if self.feature_names else sorted(feature_row.keys())
        )
        return pd.DataFrame(
            [{column: float(feature_row.get(column, 0.0)) for column in columns}]
        )

    def predict_demand(
        self,
        dt: datetime,
        actual_demand: float,
        demand_history: deque[float],
    ) -> float:
        """Predict next-step demand; fallback to actual demand when unavailable."""
        if self.model is None or not self.feature_names:
            return float(max(settings.DEMO_FORECAST_MIN_DEMAND_KWH, actual_demand))

        try:
            X = self._build_feature_row(dt, actual_demand, demand_history)
            model_scale_prediction = float(self.model.predict(X)[0])
            prediction = model_scale_prediction / self.model_to_demo_scale
            return float(max(settings.DEMO_FORECAST_MIN_DEMAND_KWH, prediction))
        except Exception as exc:
            logger.warning("Realtime demand prediction failed: %s", exc)
            noise = np.random.normal(1.0, settings.DEMO_FALLBACK_NOISE_STD)
            fallback = actual_demand * noise
            return float(max(settings.DEMO_FORECAST_MIN_DEMAND_KWH, fallback))


def get_segment_model_path(segment_name: str) -> Path:
    """Compute and validate path to the champion model for a segment.

    Args:
        segment_name: Name of the segment (e.g., 'Residential_Solar').

    Returns:
        Path to the model file.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    # Try simple naming first, then full naming as fallback.
    candidates = [
        Path("models") / f"{segment_name}_best.joblib",
        Path("models") / f"{segment_name}_LightGBM_best.joblib",
        Path("models") / f"{segment_name}_XGBoost_best.joblib",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No champion model found for segment '{segment_name}'. "
        f"Checked: {[str(c) for c in candidates]}"
    )


def load_model(path: str) -> object:
    """Load a serialised sklearn / compatible model from `path`."""
    return joblib.load(path)


def predict(model, df: pd.DataFrame) -> pd.Series:
    """Run inference on `df` using a trained forecasting model.

    Args:
        model: Trained forecasting model.
        df: DataFrame with features required by the model.

    Returns:
        Series of predictions (one per row).
    """
    from src.forecaster.modeling.train_model import get_features_and_target

    X, _ = get_features_and_target(df)
    return model.predict(X)


def get_forecast_data(
    window_size: int = 48,
    start_date: datetime | None = None,
    segment_name: str | None = None,
) -> pd.DataFrame:
    """Return market data with price, actual_demand, and predicted_demand.

    This function generates ground-truth demand from the physical model
    and enhances it with a trained ML forecaster's prediction to simulate
    realistic forecasting errors.

    The environment will:
    - Use predicted_demand in agent observations (imperfect foresight)
    - Use actual_demand for battery physics (ground truth consumption)

    Args:
        window_size: Number of hourly steps needed (typically `max_steps`).
        start_date: Start of the window; defaults to now.
        segment_name: Segment for model selection; defaults to DEFAULT_SEGMENT.

    Returns:
        DataFrame with columns `price`, `actual_demand`, `predicted_demand`,
        one row per hour, indexed 0..window_size-1.
    """
    if start_date is None:
        start_date = datetime.now()

    if segment_name is None:
        segment_name = DEFAULT_SEGMENT

    # Generate ground-truth market data
    df = generate_hourly_timeline(start_date, num_hours=window_size)

    # Rename the generated demand to actual_demand
    df = df.rename(columns={"demand": "actual_demand"})

    # Load and apply the trained forecasting model
    try:
        model_path = get_segment_model_path(segment_name)
        model = load_model(str(model_path))
        predicted_values = predict(model, df)
        df["predicted_demand"] = predicted_values
        logger.debug(
            f"Loaded champion model from {model_path} for segment {segment_name}"
        )
    except FileNotFoundError as e:
        logger.warning(
            f"Could not load model for segment {segment_name}: {e}. "
            f"Using noisy actual_demand as predicted_demand."
        )
        noise = np.random.normal(1.0, 0.08, size=len(df))
        df["predicted_demand"] = np.clip(df["actual_demand"] * noise, 0.0, None)
    except Exception as e:
        logger.warning(
            f"Model inference failed for segment {segment_name}: {e}. "
            f"Using noisy actual_demand as predicted_demand."
        )
        noise = np.random.normal(1.0, 0.08, size=len(df))
        df["predicted_demand"] = np.clip(df["actual_demand"] * noise, 0.0, None)

    # Return only the columns the environment needs
    return df[["price", "actual_demand", "predicted_demand"]].reset_index(drop=True)
