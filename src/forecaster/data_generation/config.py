from typing import Any, Dict

import pytz

TZ = pytz.timezone("America/Los_Angeles")

TRACKING_FILE = "Docs/data_generation_tracking.json"

DEFAULT_OUTPUT_CSV = "data/san_diego_energy_load_data.csv"

LOAD_PROFILES: Dict[str, Dict[str, list]] = {
    "Residential": {
        "non_solar": [
            "RES-TOU-A",
            "RES-TOU-B",
            "RES-TOU-C",
            "RES-TIER-1",
            "RES-TIER-2",
            "RES-EV-A",
            "RES-EV-B",
            "RES-LOW-INCOME",
            "RES-BASELINE",
            "RES-CARE",
            "RES-FERA",
            "RES-MEDICAL",
            "RES-TOU-D",
            "RES-TOU-E",
            "RES-SUMMER",
            "RES-WINTER",
            "RES-FLAT",
            "RES-PEAK",
            "RES-OFF-PEAK",
            "RES-STANDARD",
        ],
        "solar": [
            "RES-NEM-A",
            "RES-NEM-B",
            "RES-NEM-C",
            "RES-NEM-TOU-A",
            "RES-NEM-TOU-B",
            "RES-NEM-TOU-C",
            "RES-NEM-EV-A",
            "RES-NEM-EV-B",
            "RES-NEM-TIER-1",
            "RES-NEM-TIER-2",
            "RES-NEM-LOW-INCOME",
            "RES-NEM-CARE",
            "RES-NEM-FERA",
            "RES-NEM-MEDICAL",
            "RES-NEM-TOU-D",
            "RES-NEM-TOU-E",
            "RES-NEM-SUMMER",
            "RES-NEM-WINTER",
            "RES-NEM-PEAK",
            "RES-NEM-OFF-PEAK",
            "RES-NEM-STANDARD",
            "RES-NEM-BATTERY-A",
            "RES-NEM-BATTERY-B",
            "RES-NEM-BATTERY-C",
            "RES-NEM-FLEX",
            "RES-NEM-TIME-SHIFT",
            "RES-NEM-DEMAND-RESP",
            "RES-NEM-GRID-SUPPORT",
            "RES-NEM-MICRO-GRID",
            "RES-NEM-COMMUNITY",
            "RES-NEM-VIRTUAL-A",
            "RES-NEM-VIRTUAL-B",
            "RES-NEM-VIRTUAL-C",
            "RES-NEM-STORAGE",
            "RES-NEM-SMART-HOME",
            "RES-NEM-GREEN-TARIFF",
            "RES-NEM-ECO-FRIENDLY",
            "RES-NEM-SUSTAINABLE",
            "RES-NEM-RENEWABLE",
            "RES-NEM-CLEAN-ENERGY",
            "RES-NEM-ZERO-NET",
        ],
    },
    "Small Scale Industries": {
        "non_solar": ["SSI-GS-1", "SSI-GS-2", "SSI-TOU-8"],
        "solar": [
            "SSI-NEM-GS-1",
            "SSI-NEM-GS-2",
            "SSI-NEM-TOU-8",
            "SSI-NEM-DEMAND",
            "SSI-NEM-PEAK",
            "SSI-NEM-OFF-PEAK",
            "SSI-NEM-STORAGE",
        ],
    },
    "Medium Scale Industries": {
        "non_solar": [
            "MSI-GS-1",
            "MSI-GS-2",
            "MSI-GS-3",
            "MSI-TOU-8",
            "MSI-TOU-GS-1",
            "MSI-TOU-GS-2",
            "MSI-TOU-GS-3",
            "MSI-DEMAND-A",
            "MSI-DEMAND-B",
            "MSI-INDUSTRIAL",
        ],
        "solar": [
            "MSI-NEM-GS-1",
            "MSI-NEM-GS-2",
            "MSI-NEM-GS-3",
            "MSI-NEM-TOU-8",
            "MSI-NEM-TOU-GS-1",
            "MSI-NEM-TOU-GS-2",
            "MSI-NEM-TOU-GS-3",
            "MSI-NEM-DEMAND-A",
            "MSI-NEM-DEMAND-B",
            "MSI-NEM-INDUSTRIAL",
            "MSI-NEM-STORAGE-A",
            "MSI-NEM-STORAGE-B",
            "MSI-NEM-PEAK",
            "MSI-NEM-OFF-PEAK",
            "MSI-NEM-GRID-SUPPORT",
            "MSI-NEM-MICRO-GRID",
            "MSI-NEM-DEMAND-RESP",
            "MSI-NEM-LOAD-SHIFT",
            "MSI-NEM-RENEWABLE",
            "MSI-NEM-SUSTAINABLE",
        ],
    },
}

BASE_CONSUMPTION: Dict[str, Dict[str, Any]] = {
    "Residential": {"avg": 1.2, "std": 0.3, "meters_per_group": 2500},
    "Small Scale Industries": {"avg": 15.0, "std": 5.0, "meters_per_group": 150},
    "Medium Scale Industries": {"avg": 125.0, "std": 40.0, "meters_per_group": 25},
}

SOLAR_CAPACITY: Dict[str, Dict[str, float]] = {
    "Residential": {"avg": 7.5, "std": 2.0},
    "Small Scale Industries": {"avg": 50.0, "std": 15.0},
    "Medium Scale Industries": {"avg": 250.0, "std": 75.0},
}

LOSS_FACTORS: Dict[str, float] = {
    "Residential": 0.06,
    "Small Scale Industries": 0.04,
    "Medium Scale Industries": 0.03,
}
