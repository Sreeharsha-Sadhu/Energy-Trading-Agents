"""Unified script to train, tune, and compare LightGBM and XGBoost forecaster models.

This script:
1. Iterates over all 6 load segments in data/segments/
2. Trains LightGBM and XGBoost models using TimeSeriesSplit cross-validation
3. Selects the champion model per segment based on metric (WAPE for Solar, MAPE for Non-Solar)
4. Saves champion models with dual naming: {segment}_best.joblib and {segment}_{model}_best.joblib
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.forecaster.modeling.model_definitions import (
    get_lightgbm_model,
    get_xgboost_model,
)
from src.forecaster.modeling.train_model import (
    cross_val_score_timeseries,
    get_features_and_target,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SEGMENTS_DIR = Path("data/segments")
MODELS_DIR = Path("models")
EXPECTED_SEGMENTS = [
    "Residential_Solar",
    "Residential_Non-Solar",
    "Small_Scale_Industries_Solar",
    "Small_Scale_Industries_Non-Solar",
    "Medium_Scale_Industries_Solar",
    "Medium_Scale_Industries_Non-Solar",
]


def _filter_dataframe_by_year_range(
    df: pd.DataFrame,
    segment_name: str,
    start_year: int | None,
    end_year: int | None,
) -> pd.DataFrame:
    """Filter segment data to an inclusive year range using `Timestamp`."""
    if start_year is None and end_year is None:
        return df

    if "Timestamp" not in df.columns:
        raise ValueError(
            f"Segment {segment_name} does not contain required column 'Timestamp'"
        )

    df_filtered = df.copy()
    df_filtered["Timestamp"] = pd.to_datetime(df_filtered["Timestamp"], errors="coerce")
    df_filtered = df_filtered.dropna(subset=["Timestamp"])

    year_mask = pd.Series(True, index=df_filtered.index)
    if start_year is not None:
        year_mask &= df_filtered["Timestamp"].dt.year >= start_year
    if end_year is not None:
        year_mask &= df_filtered["Timestamp"].dt.year <= end_year

    filtered = df_filtered.loc[year_mask].copy()
    logger.info(
        "Filtered %s rows for %s -> %s rows using years [%s, %s]",
        len(df),
        segment_name,
        len(filtered),
        start_year if start_year is not None else "-inf",
        end_year if end_year is not None else "inf",
    )
    return filtered


def get_ordered_segments() -> list[str]:
    """Return ordered list of segments, filtering for those that exist."""
    segments = []
    for seg in EXPECTED_SEGMENTS:
        seg_path = SEGMENTS_DIR / f"{seg}.csv"
        if seg_path.exists():
            segments.append(seg)
        else:
            logger.warning(f"Segment file not found: {seg_path}")
    if not segments:
        raise FileNotFoundError(
            f"No segment files found in {SEGMENTS_DIR}. "
            f"Expected one or more of: {EXPECTED_SEGMENTS}"
        )
    return segments


def train_and_compare_segment(
    segment_name: str,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[str, str, float]:
    """Train and compare models for a single segment.

    Args:
        segment_name: Name of the segment (e.g., 'Residential_Solar').

    Returns:
        Tuple of (segment_name, champion_model_type, champion_metric).
    """
    seg_path = SEGMENTS_DIR / f"{segment_name}.csv"
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing segment: {segment_name}")
    logger.info(f"{'=' * 60}")

    if not seg_path.exists():
        logger.error(f"Segment file not found: {seg_path}")
        raise FileNotFoundError(f"Segment file not found: {seg_path}")

    df = pd.read_csv(seg_path)
    df = _filter_dataframe_by_year_range(df, segment_name, start_year, end_year)
    if df.empty:
        raise ValueError(
            f"No data left for segment {segment_name} after year filtering "
            f"[{start_year}, {end_year}]"
        )
    logger.info(f"Loaded {len(df)} rows from {seg_path}")

    # Determine metric based on segment type
    is_solar = segment_name.endswith("_Solar")
    metric_name = "WAPE" if is_solar else "MAPE"
    logger.info(f"Using metric: {metric_name}")

    # Train and evaluate both models
    results = {}
    for model_type in ["LightGBM", "XGBoost"]:
        logger.info(f"\nTraining {model_type}...")
        try:
            X, y = get_features_and_target(df)
            logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

            if model_type == "LightGBM":
                model = get_lightgbm_model()
            else:
                model = get_xgboost_model()

            # TimeSeriesSplit cross-validation
            cv_score = cross_val_score_timeseries(model, X, y, segment_name, splits=5)
            logger.info(f"{model_type} CV {metric_name}: {cv_score:.4f}")

            # Train final model on full data
            model.fit(X, y)

            # Save model artifact
            MODELS_DIR.mkdir(parents=True, exist_ok=True)

            # Dual naming: both formats
            filename_simple = f"{segment_name}_best.joblib"
            filename_full = f"{segment_name}_{model_type}_best.joblib"

            path_simple = MODELS_DIR / filename_simple
            path_full = MODELS_DIR / filename_full

            joblib.dump(model, path_simple)
            joblib.dump(model, path_full)
            logger.info(f"Saved: {path_simple} and {path_full}")

            results[model_type] = {
                "cv_score": cv_score,
                "model": model,
                "path_simple": path_simple,
                "path_full": path_full,
            }

        except Exception as e:
            logger.error(f"Error training {model_type}: {e}", exc_info=True)
            raise

    # Determine champion
    champion_model = min(results, key=lambda m: results[m]["cv_score"])
    champion_metric = results[champion_model]["cv_score"]

    logger.info(f"\n{'=' * 60}")
    logger.info(
        f"CHAMPION: {champion_model} with {metric_name} = {champion_metric:.4f}"
    )
    logger.info(f"{'=' * 60}")

    for model_type, res in results.items():
        status = "✓ CHAMPION" if model_type == champion_model else "  runner-up"
        logger.info(f"  {status} {model_type}: {metric_name} = {res['cv_score']:.4f}")

    return segment_name, champion_model, champion_metric


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare LightGBM/XGBoost models across all segments"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Inclusive start year for training data filter (example: 2022)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Inclusive end year for training data filter (example: 2024)",
    )
    return parser.parse_args()


def main():
    """Execute the full training and model comparison pipeline."""
    args = _parse_args()

    if (
        args.start_year is not None
        and args.end_year is not None
        and args.start_year > args.end_year
    ):
        raise ValueError("--start-year cannot be greater than --end-year")

    logger.info("Starting model training and comparison pipeline...")
    logger.info(
        "Training year filter: start=%s end=%s",
        args.start_year,
        args.end_year,
    )

    try:
        segments = get_ordered_segments()
        logger.info(f"Found {len(segments)} segment(s) to process")
    except FileNotFoundError as e:
        logger.error(f"Failed to discover segments: {e}")
        return False

    results = []
    failed = []

    for segment in segments:
        try:
            seg_name, champion, metric = train_and_compare_segment(
                segment,
                start_year=args.start_year,
                end_year=args.end_year,
            )
            results.append((seg_name, champion, metric))
        except Exception as e:
            logger.error(f"Failed to process segment {segment}: {e}")
            failed.append(segment)

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Successfully processed: {len(results)}/{len(segments)} segments")

    if results:
        logger.info("\nChampion models selected:")
        for seg_name, champion, metric in sorted(results):
            logger.info(f"  {seg_name}: {champion} ({metric:.4f})")

    if failed:
        logger.warning(f"\nFailed segments: {', '.join(failed)}")
        return False

    logger.info(f"\n✓ Pipeline complete. Model artifacts saved to {MODELS_DIR}/")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
