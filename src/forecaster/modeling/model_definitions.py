import lightgbm as lgb
import xgboost as xgb
from .deep_models import PyTorchScikitWrapper, HybridLSTMCNN, KalmanVikingModel

def get_xgboost_model(params=None):
    """Get Xgboost Model."""
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
    """Get Lightgbm Model."""
    default_params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "max_depth": -1,
        "n_estimators": 300,
    }
    if params:
        default_params.update(params)
    return lgb.LGBMRegressor(**default_params)


def get_lstm_cnn_model(params=None):
    """Get PyTorch Hybrid LSTM-CNN Model."""
    # input_dim will be resolved at fit time
    return PyTorchScikitWrapper(model_class=HybridLSTMCNN, input_dim=1, epochs=10, lr=0.001)


def get_kalman_viking_model(params=None):
    """Get PyTorch Kalman-Viking Model."""
    # input_dim will be resolved at fit time
    return PyTorchScikitWrapper(model_class=KalmanVikingModel, input_dim=1, epochs=10, lr=0.001)
