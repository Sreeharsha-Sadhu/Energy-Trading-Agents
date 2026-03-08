"""Data provider for the live demo.
Wraps the existing forecaster's EnergyLoadDataGenerator to generate and
serve time-series (price, demand) tuples for the simulation loop.
"""

import logging
from datetime import datetime, timedelta
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd

from src.forecaster.data_generation.config import BASE_CONSUMPTION
from src.forecaster.data_generation.load_factors import (
    get_hourly_load_factor,
    get_seasonal_factor,
    get_weekend_factor,
)
from src.forecaster.data_generation.solar_model import get_solar_irradiance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _compute_spot_price(hour: int, month: int, weekday: int, solar_irr: float) -> float:
    """Derive a realistic spot price ($/kWh) using load factors and solar supply.
    Higher demand + lower solar → higher price.
    """
    base_price = 0.12  # $/kWh off-peak baseline

    demand_mult = get_hourly_load_factor(hour, "Residential")
    seasonal_mult = get_seasonal_factor(month)
    weekend_mult = get_weekend_factor(weekday)

    solar_discount = solar_irr * 0.08  # max ~8c/kWh discount at peak sun

    price = base_price * demand_mult * seasonal_mult * weekend_mult - solar_discount
    price *= np.random.normal(1.0, 0.03)
    return float(max(0.02, round(price, 4)))


def _compute_demand(hour: int, month: int, weekday: int) -> float:
    """Compute aggregated demand (kWh) for a single time step using the
    existing residential load factors.  Scaled to be in the 1-10 kWh range
    so it interacts meaningfully with the 50 kWh battery env.
    """
    base = BASE_CONSUMPTION["Residential"]["avg"]  # 1.2 kWh per meter
    hour_f = get_hourly_load_factor(hour, "Residential")
    season_f = get_seasonal_factor(month)
    weekend_f = get_weekend_factor(weekday)

    demand = base * hour_f * season_f * weekend_f
    demand *= np.random.normal(1.0, 0.05)
    return float(max(0.1, round(demand, 3)))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def generate_hourly_timeline(
    start_date: datetime,
    num_hours: int = 168,  # 7 days default
) -> pd.DataFrame:
    """Generate an hourly DataFrame of (datetime, price, demand) rows using
    the forecaster's physical models.
    """
    rows: List[dict] = []
    for h in range(num_hours):
        dt = start_date + timedelta(hours=h)
        solar = get_solar_irradiance(dt)
        price = _compute_spot_price(dt.hour, dt.month, dt.weekday(), solar)
        demand = _compute_demand(dt.hour, dt.month, dt.weekday())
        rows.append({"datetime": dt, "price": price, "demand": demand})

    return pd.DataFrame(rows)


def iterate_market_ticks(
    start_date: datetime,
    num_hours: int = 168,
) -> Generator[Tuple[datetime, float, float], None, None]:
    """Yield (datetime, price, demand) tuples one-by-one, simulating a
    real-time market feed.  The simulation loop calls this lazily.
    """
    for h in range(num_hours):
        dt = start_date + timedelta(hours=h)
        solar = get_solar_irradiance(dt)
        price = _compute_spot_price(dt.hour, dt.month, dt.weekday(), solar)
        demand = _compute_demand(dt.hour, dt.month, dt.weekday())
        yield dt, price, demand
