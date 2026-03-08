"""Tests for the demo data provider module."""

from datetime import datetime

import pandas as pd

from src.demo.data_provider import (
    _compute_demand,
    _compute_spot_price,
    generate_hourly_timeline,
    iterate_market_ticks,
)


def test_spot_price_is_positive():
    price = _compute_spot_price(hour=14, month=7, weekday=2, solar_irr=0.5)
    assert price > 0
    assert isinstance(price, float)


def test_demand_is_positive():
    demand = _compute_demand(hour=18, month=1, weekday=0)
    assert demand > 0
    assert isinstance(demand, float)


def test_demand_higher_during_peak_hours():
    """Evening demand (hour 17) should generally be higher than early morning (hour 3)."""
    evening_demands = [_compute_demand(17, 7, 2) for _ in range(50)]
    morning_demands = [_compute_demand(3, 7, 2) for _ in range(50)]
    assert sum(evening_demands) / len(evening_demands) > sum(morning_demands) / len(
        morning_demands
    )


def test_generate_hourly_timeline_shape():
    start = datetime(2025, 3, 7, 0, 0)
    df = generate_hourly_timeline(start, num_hours=48)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 48
    assert set(df.columns) == {"datetime", "price", "demand"}
    assert (df["price"] > 0).all()
    assert (df["demand"] > 0).all()


def test_iterate_market_ticks_count():
    start = datetime(2025, 3, 7, 0, 0)
    ticks = list(iterate_market_ticks(start, num_hours=24))
    assert len(ticks) == 24
    for dt, price, demand in ticks:
        assert price > 0
        assert demand > 0
