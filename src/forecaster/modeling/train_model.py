import os

import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .metrics import mape, wape
from .model_definitions import get_lightgbm_model, get_xgboost_model

MODELS_DIR = "models/"


def get_features_and_target(df: pd.DataFrame, target="BaseLoad"):
    """Get Features And Target."""
    drop_cols = [
        "TradeDate",
        "TradeTime",
        "LoadProfile",
        "RateGroup",
        "Submission",
        "LossAdjustedLoad",
        "LoadBL",
        "LoadLAL",
        "GenBL",
        "GenLAL",
        "Solar_Status",
        "Created",
    ]
    features = [c for c in df.columns if c not in drop_cols + [target]]
    return df[features], df[target]


def cross_val_score_timeseries(model, X, y, segment_name: str, splits: int = 5):
    """Perform TimeSeriesSplit CV and return average metric."""
    tscv = TimeSeriesSplit(n_splits=splits)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metric = wape(y_test, preds) if "Solar" in segment_name else mape(y_test, preds)
        scores.append(metric)

    return sum(scores) / len(scores)


def train_single_model(df: pd.DataFrame, segment_name: str, model_type: str):
    """Train using TimeSeriesSplit CV, then fit final model on full data."""
    X, y = get_features_and_target(df)

    if model_type == "XGBoost":
        model = get_xgboost_model()
    else:
        model = get_lightgbm_model()

    cv_score = cross_val_score_timeseries(model, X, y, segment_name)

    model.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    filename = f"{segment_name}_{model_type}_best.joblib"
    joblib.dump(model, os.path.join(MODELS_DIR, filename))

    return cv_score, filename
