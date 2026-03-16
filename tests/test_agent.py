import os

import numpy as np

from src.agent.ppo_model import predict_action, train_agent
from src.config import settings


def test_training_and_prediction():
    original_timesteps = settings.TOTAL_TIMESTEPS
    settings.TOTAL_TIMESTEPS = 100

    try:
        model = train_agent()
        assert model is not None
        assert os.path.exists(settings.MODEL_SAVE_PATH)

        obs = np.array([0.5, 2.0, 10.0, 100.0], dtype=np.float32)
        action = predict_action(obs)
        assert isinstance(action, float)
        assert -1.0 <= action <= 1.0
    finally:
        settings.TOTAL_TIMESTEPS = original_timesteps
