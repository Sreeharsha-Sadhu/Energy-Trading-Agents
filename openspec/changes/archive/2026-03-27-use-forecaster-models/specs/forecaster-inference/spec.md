# Spec: forecaster-inference

## Description
Modifications to `inference.py` and the RL environment to decouple `actual_demand` from `predicted_demand` using a trained `.joblib` model.

## Requirements
- `get_forecast_data` MUST return a DataFrame with `price`, `actual_demand`, and `predicted_demand`.
- `get_forecast_data` MUST load a pre-trained `.joblib` model to generate `predicted_demand`.
- `EnergyTradingEnv` MUST provide `predicted_demand` as the observation to the agent.
- `EnergyTradingEnv` MUST deduct `actual_demand` from the battery.

## Scenarios
- **GIVEN** a trained forecaster model
- **WHEN** the agent steps through the environment
- **THEN** the observation state uses the model's prediction, but the physical battery level is affected by the actual simulated load.
