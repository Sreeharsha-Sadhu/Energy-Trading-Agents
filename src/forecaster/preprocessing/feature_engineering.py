"""Feature engineering reimplemented from pre-processing.py:
- time features
- simulated weather (Temperature, Solar_Irradiance)
- lag features and rolling windows (grouped by LoadProfile & RateGroup).
"""

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Time Features."""
    df = df.copy()
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek  # 0 = Monday
    df["Month"] = df["Timestamp"].dt.month
    df["Quarter"] = df["Timestamp"].dt.quarter
    df["Year"] = df["Timestamp"].dt.year
    df["DayOfYear"] = df["Timestamp"].dt.dayofyear
    df["WeekOfYear"] = df["Timestamp"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    return df


def simulate_weather(df_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Recreate the synthetic weather generation logic from the notebook:
    - Temperature (F)
    - Solar_Irradiance (W/m^2-like scale, clipped >= 0).
    """
    # `unique()` can return DatetimeArray on newer pandas; normalize to index.
    idx = pd.DatetimeIndex(df_index.unique()).sort_values()
    base_temp = 65  # Fahrenheit baseline
    dayofyear = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()

    seasonal_effect = -15 * np.cos(2 * np.pi * (dayofyear - 30) / 365.25)
    diurnal_effect = -5 * np.cos(2 * np.pi * hour / 24)
    noise = np.random.normal(0, 2, size=len(idx))

    temp = base_temp + seasonal_effect + diurnal_effect + noise

    solar_seasonal = 0.5 * (1 - np.cos(2 * np.pi * (dayofyear - 172) / 365.25))
    solar_diurnal = np.maximum(0, np.cos(2 * np.pi * (hour - 12) / 24)) ** 2
    solar_irrad = (solar_seasonal * solar_diurnal * 1000) + np.random.normal(
        0, 20, size=len(idx)
    )
    solar_irrad = np.clip(solar_irrad, 0, None)

    weather_df = pd.DataFrame(index=idx)
    weather_df["Temperature"] = temp
    weather_df["Solar_Irradiance"] = solar_irrad
    return weather_df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lag and rolling features grouped by LoadProfile & RateGroup
    - lag 168h (1 week)
    - 7-day rolling mean based on the lag field to avoid leakage.
    """
    df = df.copy()
    grouped = df.sort_values("Timestamp").groupby(["LoadProfile", "RateGroup"])
    df["BaseLoad_lag_168h"] = grouped["BaseLoad"].shift(168)
    df["BaseLoad_rolling_mean_7d"] = (
        df["BaseLoad_lag_168h"].rolling(window=7 * 24, min_periods=1).mean()
    )
    df = df.dropna().reset_index(drop=True)
    return df
