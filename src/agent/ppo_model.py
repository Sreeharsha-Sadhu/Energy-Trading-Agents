import os

import numpy as np
from stable_baselines3 import PPO

from src.config import settings
from src.envs.energy_trading_env import EnergyTradingEnv


def train_agent(env=None):
    """Train Agent."""
    if env is None:
        env = EnergyTradingEnv()

    model = PPO("MlpPolicy", env, verbose=1)

    os.makedirs(os.path.dirname(settings.MODEL_SAVE_PATH), exist_ok=True)

    model.learn(total_timesteps=settings.TOTAL_TIMESTEPS)
    model.save(settings.MODEL_SAVE_PATH)
    return model


def predict_action(obs: np.ndarray) -> int:
    """Predict Action."""
    if not os.path.exists(settings.MODEL_SAVE_PATH):
        return 2

    model = PPO.load(settings.MODEL_SAVE_PATH)
    action, _states = model.predict(obs, deterministic=True)
    return int(action)
