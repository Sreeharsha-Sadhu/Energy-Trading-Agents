from src.agent.ppo_model import train_agent, predict_action
from src.config import settings
import os
import numpy as np


def test_training_and_prediction():
    # temporarily reduce timesteps for quick test
    original_timesteps = settings.TOTAL_TIMESTEPS
    settings.TOTAL_TIMESTEPS = 100

    try:
        model = train_agent()
        assert model is not None
        assert os.path.exists(settings.MODEL_SAVE_PATH)

        obs = np.array([0.5, 2.0, 10.0, 100.0], dtype=np.float32)
        action = predict_action(obs)
        assert action in [0, 1, 2]
    finally:
        settings.TOTAL_TIMESTEPS = original_timesteps
