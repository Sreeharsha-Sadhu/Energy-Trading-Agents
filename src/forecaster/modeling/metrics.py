import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred):
    """Rmse."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Mae."""
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    """Mape."""
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def wape(y_true, y_pred):
    """Wape."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
