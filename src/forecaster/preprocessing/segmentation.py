"""Functions to create and save modeling segments (six segments: Residential/Small/Medium x Solar/Non-Solar)."""

import os
from typing import Dict

import pandas as pd


def add_solar_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add Solar Status."""
    df = df.copy()
    df["Solar_Status"] = df["RateGroup"].apply(
        lambda x: "Solar" if "NEM" in str(x) else "Non-Solar"
    )
    return df


def create_segments(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create Segments."""
    segments = {}
    for profile in df["LoadProfile"].unique():
        for status in ["Non-Solar", "Solar"]:
            segment_name = f"{profile.replace(' ', '_')}_{status}"
            seg_df = df[
                (df["LoadProfile"] == profile) & (df["Solar_Status"] == status)
            ].copy()
            segments[segment_name] = seg_df
    return segments


def save_segments(
    segments: Dict[str, pd.DataFrame], out_dir: str = "data/segments/"
) -> None:
    """Save Segments."""
    os.makedirs(out_dir, exist_ok=True)
    for name, seg in segments.items():
        fp = os.path.join(out_dir, f"{name}.csv")
        seg.to_csv(fp, index=True)
