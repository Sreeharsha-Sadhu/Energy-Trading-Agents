"""Cleaning utilities converted from pre-processing.py
- Timestamp creation
- Sorting
- Prioritize 'Final' submissions over 'Initial'
- Basic sanity checks.
"""

import pandas as pd


def ensure_timestamp(
    df: pd.DataFrame,
    date_col: str = "TradeDate",
    time_col: str = "TradeTime",
    tz_localize: bool = False,
    tz: str = None,
) -> pd.DataFrame:
    """Create 'Timestamp' column from TradeDate + TradeTime and sort the DF.
    If tz_localize is True, will localize to provided tz (e.g., 'America/Los_Angeles').
    """
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(
        df[date_col].astype(str) + " " + df[time_col].astype(str)
    )
    if tz_localize and tz:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize(tz)
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def prioritize_final_submissions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most accurate submission per (Timestamp, LoadProfile, RateGroup).
    The project used 'Final' vs 'Initial' where 'Final' is preferred.
    This function keeps the final submission record if duplicates exist.
    """
    df = df.copy()
    rank_map = {"Final": 1, "Initial": 0}
    df["_submission_rank"] = df["Submission"].map(rank_map).fillna(0).astype(int)
    df = df.sort_values(
        ["Timestamp", "LoadProfile", "RateGroup", "_submission_rank"],
        ascending=[True, True, True, False],
    )
    df = df.drop_duplicates(
        subset=["Timestamp", "LoadProfile", "RateGroup"], keep="first"
    )
    df.drop(columns=["_submission_rank"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
