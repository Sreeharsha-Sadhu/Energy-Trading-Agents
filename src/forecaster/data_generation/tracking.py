import json
import os
from typing import Any, Dict


def load_tracking_data(tracking_file: str) -> Dict[str, Any]:
    """Load Tracking Data."""
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, "r") as f:
                return json.load(f)
        except Exception:
            return {
                "last_initial_date": None,
                "last_final_date": None,
                "first_run": True,
            }
    return {"last_initial_date": None, "last_final_date": None, "first_run": True}


def save_tracking_data(tracking_file: str, data: Dict[str, Any]) -> None:
    """Save Tracking Data."""
    os.makedirs(os.path.dirname(tracking_file) or ".", exist_ok=True)
    with open(tracking_file, "w") as f:
        json.dump(data, f, indent=2)
