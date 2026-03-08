import math
from datetime import datetime
from typing import Optional

import numpy as np


def get_solar_irradiance(
    dt: datetime,
    lat: float = 32.7157,
    lon: float = -117.1611,
    seed: Optional[int] = None,
) -> float:
    """Approximate normalized solar irradiance [0,1] for a datetime and lat/lon.
    Simple physical model using declination and hour angle with a small weather noise.
    """
    day_of_year = dt.timetuple().tm_yday
    hour = dt.hour

    declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365.0))
    hour_angle = 15 * (hour - 12)

    elevation = math.asin(
        math.sin(math.radians(declination)) * math.sin(math.radians(lat))
        + math.cos(math.radians(declination))
        * math.cos(math.radians(lat))
        * math.cos(math.radians(hour_angle))
    )

    if elevation <= 0:
        return 0.0

    base_irradiance = math.sin(elevation)
    if seed is not None:
        rng = np.random.RandomState(seed)
        weather_factor = rng.normal(0.85, 0.15)
    else:
        weather_factor = np.random.normal(0.85, 0.15)

    return max(0.0, min(1.0, base_irradiance * weather_factor))
