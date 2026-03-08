import os

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .metrics import mape, wape
from .model_definitions import get_lightgbm_model, get_xgboost_model
from .train_model import get_features_and_target

TUNING_DB = "tuning.db"
TUNING_CSV = "tuning_results.csv"
MODELS_DIR = "models/"


def cv_objective(trial, X, y, segment_name, model_type):
    """CV objective using TimeSeriesSplit → realistic forecasting evaluation."""
    if model_type == "LightGBM":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", -1, 20),
        }
        model = get_lightgbm_model(params)

    else:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        }
        model = get_xgboost_model(params)

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metric = wape(y_test, preds) if "Solar" in segment_name else mape(y_test, preds)
        scores.append(metric)

    return float(np.mean(scores))


def tune_segment_model(segment_name: str, df: pd.DataFrame, model_type="LightGBM"):
    """Tune Segment Model."""
    X, y = get_features_and_target(df)

    def objective(trial):
        """Objective."""
        return cv_objective(trial, X, y, segment_name, model_type)

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{TUNING_DB}",
        study_name=segment_name,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=40)

    best_params = study.best_params

    if model_type == "LightGBM":
        model = get_lightgbm_model(best_params)
    else:
        model = get_xgboost_model(best_params)

    model.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_file = os.path.join(MODELS_DIR, f"{segment_name}_{model_type}_best.joblib")
    joblib.dump(model, out_file)

    return best_params, out_file
