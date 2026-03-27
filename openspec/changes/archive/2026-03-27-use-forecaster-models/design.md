# Design: use-forecaster-models

## Architecture / Approach
Our approach centers on automating the identification of the best machine learning forecaster and plugging its predictions directly into the `EnergyTradingEnv`. 

1. **Selection Pipeline (`scripts/train_and_compare_models.py`)**: 
   We will combine the logic from `train_model.py`, `tune_model.py`, and `compare_models.py` into a unified script. This script will iterate over all 6 segments, train and tune LightGBM & XGBoost models, run `TimeSeriesSplit` cross-validation, and overwrite the segment's `{segment}_best.joblib` artifact with the best performing model based on MAPE/WAPE.
2. **Realistic Inference (`src/forecaster/modeling/inference.py`)**:
   We will update `get_forecast_data` to load the `Residential_Solar_LightGBM_best.joblib` (or an equivalent default champion model). The script will generate the ground truth realistic load data using `data_provider.py`, run inference using the loaded `.joblib` model, and provide the environment with both the `actual_demand` (for battery physics) and the `predicted_demand` (for agent observation).

## Data Model / Schema Changes
No database schema changes. State representation in the Gym environment will remain identical, however, the meaning of `forecasted_demand` will change from perfect ground truth to noisy predictions.

## API / Interface Changes
- `src/envs/energy_trading_env.py` will expect `get_forecast_data()` to return a DataFrame containing both `actual_demand` and `predicted_demand`.
- `EnergyTradingEnv.step()` will subtract `actual_demand` from the battery.

## Out of Scope
- Full retraining of the PPO agent is out of scope for the data pipeline integration step, though it may perform worse initially due to imperfect forecasts.

## Decisions
- Used existing feature generation and pre-trained `.joblib` model loading rather than querying an external prediction API to keep the simulation self-contained.

## Risks / Trade-offs
- Model drift or high forecasting error could severely penalize the RL agent. We may eventually need to tune the variance penalty in the reward function if the predicting model has high error.
