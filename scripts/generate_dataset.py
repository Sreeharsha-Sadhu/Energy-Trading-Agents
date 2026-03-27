"""CLI wrapper for synthetic dataset generation."""

from __future__ import annotations

import argparse
from datetime import datetime

from src.forecaster.data_generation.energy_load_generator import EnergyLoadDataGenerator

DEFAULT_CHUNK_SIZE = 50000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic energy load dataset"
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Generate data availability as of this date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full regeneration starting from 2020-01-01.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of rows buffered before appending to CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_date: datetime | None = None
    if args.as_of_date:
        run_date = datetime.fromisoformat(args.as_of_date)

    generator = EnergyLoadDataGenerator()
    generator.generate_incremental_dataset(
        current_date=run_date,
        force_full_regeneration=args.full,
        chunk_size=args.chunk_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
