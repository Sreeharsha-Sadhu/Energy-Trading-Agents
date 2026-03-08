import numpy as np


def get_hourly_load_factor(hour: int, load_profile: str) -> float:
    """Hourly multiplier arrays copied from original logic."""
    if load_profile == "Residential":
        base_factors = [
            0.6,
            0.55,
            0.5,
            0.5,
            0.55,
            0.7,
            0.9,
            1.1,
            1.0,
            0.8,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            1.0,
            1.2,
            1.4,
            1.3,
            1.2,
            1.1,
            1.0,
            0.8,
            0.7,
        ]

    elif load_profile == "Small Scale Industries":
        base_factors = [
            0.3,
            0.2,
            0.2,
            0.2,
            0.3,
            0.5,
            0.8,
            1.2,
            1.4,
            1.3,
            1.2,
            1.1,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.2,
            0.9,
            0.7,
            0.5,
            0.4,
            0.3,
            0.3,
        ]
    else:
        base_factors = [
            0.7,
            0.6,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.2,
            1.3,
            1.3,
            1.2,
            1.1,
            1.0,
            1.1,
            1.2,
            1.3,
            1.2,
            1.1,
            1.0,
            0.9,
            0.8,
            0.8,
            0.7,
            0.7,
        ]
    return base_factors[hour]


def get_seasonal_factor(month: int) -> float:
    """Get Seasonal Factor."""
    seasonal_factors = {
        1: 0.8,
        2: 0.75,
        3: 0.8,
        4: 0.9,
        5: 1.0,
        6: 1.2,
        7: 1.4,
        8: 1.4,
        9: 1.3,
        10: 1.1,
        11: 0.9,
        12: 0.8,
    }
    return seasonal_factors.get(month, 1.0)


def get_weekend_factor(weekday: int) -> float:
    """Get Weekend Factor."""
    return 0.7 if weekday >= 5 else 1.0


def generate_meter_count(base_count: int, variation_sd: float = 0.05) -> int:
    """Generate Meter Count."""
    variation = np.random.normal(1.0, variation_sd)
    return int(max(1, base_count * variation))
