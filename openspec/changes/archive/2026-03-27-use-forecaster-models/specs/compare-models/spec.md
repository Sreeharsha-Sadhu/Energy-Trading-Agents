# Spec: compare-models

## Description
A unified script that trains, tunes, and compares LightGBM and XGBoost forecaster models for each of the 6 data segments, and saves the best model.

## Requirements
- The script MUST iterate over all 6 load segments.
- The script MUST use `TimeSeriesSplit` cross-validation for both tuning and comparison.
- The script MUST evaluate both LightGBM and XGBoost models.
- The script MUST select the champion model based on the lowest WAPE (for Solar) or MAPE (for Non-Solar).
- The script MUST export the champion model to `models/{segment}_{model}_best.joblib`.

## Scenarios
- **GIVEN** historical energy load segments
- **WHEN** the script `scripts/train_and_compare_models.py` is executed
- **THEN** 6 `.joblib` artifacts representing the best forecaster for each segment are saved to the `models/` directory.
