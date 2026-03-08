"""Cross-validation visualization utilities for time series forecasting.
Generates:
 - Fold-wise CV metric plots
 - Model comparison CV plots
 - Optuna trial vs. score plots.
"""

from typing import Dict

import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

from .metrics import mape, wape


def plot_cv_folds(model, X, y, segment_name: str, figsize=(10, 6)):
    """Plot metric on each TimeSeriesSplit fold."""
    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    metric_func = wape if "Solar" in segment_name else mape

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_metric = metric_func(y_test, preds)
        fold_metrics.append(fold_metric)

    plt.figure(figsize=figsize)
    plt.plot(range(1, len(fold_metrics) + 1), fold_metrics, marker="o")
    plt.title(f"Cross-Validation Performance per Fold\n({segment_name})")
    plt.xlabel("Fold Number")
    plt.ylabel("Metric (MAPE/WAPE)")
    plt.grid(True)
    plt.show()

    return fold_metrics


def plot_model_comparison_cv(
    model_scores: Dict[str, float], segment_name: str, figsize=(10, 6)
):
    """Bar chart comparing CV averages across models.

    Example:
    model_scores = {"LightGBM": 12.5, "XGBoost": 13.8}.

    """
    names = list(model_scores.keys())
    values = list(model_scores.values())

    plt.figure(figsize=figsize)
    bars = plt.bar(names, values)
    plt.title(f"Model Cross-Validation Comparison\n({segment_name})")
    plt.ylabel("CV Metric (Lower = Better)")
    plt.grid(axis="y")

    for bar, score in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            score,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.show()


def plot_optuna_trials(study, figsize=(10, 6)):
    """Plot Optuna trial number vs. objective value for diagnostics."""
    trials = study.trials_dataframe()
    completed = trials[trials["state"] == "COMPLETE"]

    plt.figure(figsize=figsize)
    plt.scatter(completed["number"], completed["value"], label="Trial Score")
    plt.plot(completed["number"], completed["value"], alpha=0.4)
    plt.title("Optuna Trial Scores (Objective Value)")
    plt.xlabel("Trial Number")
    plt.ylabel("Objective Metric (Lower = Better)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_optuna_param_importance(study, figsize=(10, 6)):
    """Plot parameter importance using Optuna's importance evaluator."""
    import optuna

    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
        print(
            "Importance evaluator failed — ensure you have multiple completed trials."
        )
        return

    params = list(importance.keys())
    values = list(importance.values())

    plt.figure(figsize=figsize)
    bars = plt.barh(params, values)
    plt.title("Optuna Parameter Importance")
    plt.xlabel("Relative Importance")
    plt.grid(True, axis="x")

    for bar, val in zip(bars, values):
        plt.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center"
        )

    plt.show()
