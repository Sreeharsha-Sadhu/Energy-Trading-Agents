# src/data_generation/tracking.py
import os
import json
from typing import Dict, Any


def load_tracking_data(tracking_file: str) -> Dict[str, Any]:
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, "r") as f:
                return json.load(f)
        except Exception:
            # if file corrupted, return default
            return {
                "last_initial_date": None,
                "last_final_date": None,
                "first_run": True,
            }
    return {"last_initial_date": None, "last_final_date": None, "first_run": True}


def save_tracking_data(tracking_file: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(tracking_file) or ".", exist_ok=True)
    with open(tracking_file, "w") as f:
        json.dump(data, f, indent=2)
