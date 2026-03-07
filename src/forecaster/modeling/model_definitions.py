# src/modeling/model_definitions.py

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def get_xgboost_model(params=None):
    default_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 300,
    }
    if params:
        default_params.update(params)
    return xgb.XGBRegressor(**default_params)


def get_lightgbm_model(params=None):
    default_params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "max_depth": -1,
        "n_estimators": 300,
    }
    if params:
        default_params.update(params)
    return lgb.LGBMRegressor(**default_params)
