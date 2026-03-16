import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.config import settings
from src.envs.energy_trading_env import EnergyTradingEnv


def train_agent(env=None):
    """Train Agent."""
    if env is None:
        env = make_vec_env(EnergyTradingEnv, n_envs=4)

    model = PPO("MlpPolicy", env, verbose=1)

    os.makedirs(os.path.dirname(settings.MODEL_SAVE_PATH), exist_ok=True)

    model.learn(total_timesteps=settings.TOTAL_TIMESTEPS)
    model.save(settings.MODEL_SAVE_PATH)
    return model


def predict_action(obs: np.ndarray) -> float:
    """Predict Action.

    Returns the continuous action scalar in [-1.0, 1.0]. Returns 0.0 (hold)
    if no trained model is found.
    """
    if not os.path.exists(settings.MODEL_SAVE_PATH):
        return 0.0

    model = PPO.load(settings.MODEL_SAVE_PATH)
    action, _states = model.predict(obs, deterministic=True)
    return float(action[0])
