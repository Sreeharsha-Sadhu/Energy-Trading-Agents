import joblib
import pandas as pd

from .train_model import get_features_and_target


def load_model(path: str):
    """Load Model."""
    return joblib.load(path)


def predict(model, df: pd.DataFrame):
    """Predict."""
    X, _ = get_features_and_target(df)
    return model.predict(X)
