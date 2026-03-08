import os

import joblib
import pandas as pd

from .metrics import mape, wape
from .train_model import get_features_and_target


def compare_models(segment_name: str, df: pd.DataFrame, model_files: list):
    """Load each model, score it on full df, return comparison table."""
    X, y = get_features_and_target(df)
    results = []

    for mfile in model_files:
        if not os.path.exists(mfile):
            continue
        model = joblib.load(mfile)
        preds = model.predict(X)
        metric = wape(y, preds) if "Solar" in segment_name else mape(y, preds)

        results.append({"Model_File": mfile, "Segment": segment_name, "Metric": metric})

    return pd.DataFrame(results).sort_values("Metric")
