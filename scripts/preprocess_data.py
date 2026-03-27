"""CLI wrapper for preprocessing and segmentation."""

from __future__ import annotations

import argparse

from src.forecaster.preprocessing.preprocess import run_full_preprocessing

DEFAULT_INPUT_PATH = "data/san_diego_energy_load_data.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline and regenerate segments"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Raw dataset CSV path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite processed full CSV output",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_full_preprocessing(input_path=args.input, overwrite=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
