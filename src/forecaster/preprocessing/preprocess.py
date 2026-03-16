"""Orchestrator: full pipeline to create processed dataset and segments.
Usage (programmatic):
    from src.preprocessing.preprocess import run_full_preprocessing
    run_full_preprocessing(input_path="data/san_diego_energy_load_data.csv").
"""

from pathlib import Path

import pandas as pd

from .cleaning import ensure_timestamp, prioritize_final_submissions
from .dataset_loader import load_raw_dataset
from .feature_engineering import (
    add_lag_and_rolling_features,
    add_time_features,
    simulate_weather,
)
from .segmentation import add_solar_status, create_segments, save_segments

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_FULL_PATH = PROCESSED_DIR / "processed_full.csv"


def run_full_preprocessing(
    input_path: str = "data/san_diego_energy_load_data.csv", overwrite: bool = False
) -> pd.DataFrame:
    df = load_raw_dataset(input_path)

    df = ensure_timestamp(df)
    df = prioritize_final_submissions(df)

    df = add_time_features(df)
    weather = simulate_weather(df["Timestamp"])
    df = df.merge(weather, left_on="Timestamp", right_index=True, how="left")

    df = add_lag_and_rolling_features(df)

    df = add_solar_status(df)
    segments = create_segments(df)

    if overwrite or not PROCESSED_FULL_PATH.exists():
        df.to_csv(PROCESSED_FULL_PATH, index=False)

    save_segments(segments, out_dir="data/segments/")

    return df
