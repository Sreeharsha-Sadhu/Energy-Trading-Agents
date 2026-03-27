# Tasks: use-forecaster-models

## 1. Model Selection Script

- [x] 1.1 Create `scripts/train_and_compare_models.py`.
- [x] 1.2 Implement iteration over the 6 data segments in `data/segments/`.
- [x] 1.3 Implement LightGBM and XGBoost training using `TimeSeriesSplit` cross-validation.
- [x] 1.4 Implement evaluation logic to select the best model per segment based on MAPE/WAPE.
- [x] 1.5 Implement serialization to save the best model to `models/{segment}_best.joblib`.

## 2. Realistic Inference Integration

- [x] 2.1 Modify `get_forecast_data` in `src/forecaster/modeling/inference.py` to additionally return `actual_demand` and `predicted_demand`, loading the pre-trained `.joblib` model for the forecast.
- [x] 2.2 Modify `EnergyTradingEnv.step()` and `_get_obs()` in `src/envs/energy_trading_env.py` to use `predicted_demand` for the agent's observation and `actual_demand` for battery drain logic.
