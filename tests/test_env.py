import pytest
import numpy as np
import gymnasium as gym
from src.envs.energy_trading_env import EnergyTradingEnv
from src.config import settings


@pytest.fixture
def env():
    return EnergyTradingEnv()


def test_env_initialization(env):
    obs, info = env.reset()
    assert env.observation_space.shape == (4,)
    assert env.action_space.n == 3
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32


def test_trade_sell_empty_battery():
    env = EnergyTradingEnv()
    obs, info = env.reset()

    # Force battery to 0
    env.battery_level = 0.0

    # Action 1 is Sell
    obs, reward, terminated, truncated, info = env.step(1)

    # Reward should be negative (penalty for invalid action)
    assert reward < 0
    # Battery should remain 0
    assert env.battery_level == 0.0


def test_trade_buy_insufficient_balance():
    env = EnergyTradingEnv()
    obs, info = env.reset()

    # Force balance to 0
    env.account_balance = 0.0

    # Action 0 is Buy
    obs, reward, terminated, truncated, info = env.step(0)

    # Reward should be negative (penalty)
    assert reward < 0
    # Balance should remain 0
    assert env.account_balance == 0.0


def test_trade_buy_full_battery():
    env = EnergyTradingEnv()
    obs, info = env.reset()

    # Force battery to max
    env.battery_level = settings.MAX_BATTERY_CAPACITY_KWH
    env.account_balance = 1000.0  # Enough money
    env.forecasted_demand = 0.0  # Prevent demand from reducing battery

    # Action 0 is Buy
    obs, reward, terminated, truncated, info = env.step(0)

    assert reward < 0
    assert env.battery_level == settings.MAX_BATTERY_CAPACITY_KWH


def test_valid_buy_and_sell(env):
    obs, info = env.reset()
    env.battery_level = settings.MAX_BATTERY_CAPACITY_KWH / 2
    env.account_balance = 100.0
    env.current_price = 1.0  # Set a known price

    # Buy
    action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.battery_level > settings.MAX_BATTERY_CAPACITY_KWH / 2
    assert env.account_balance < 100.0

    # Sell
    env.current_price = 2.0
    prev_balance = env.account_balance
    prev_battery = env.battery_level
    action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.battery_level < prev_battery
    assert env.account_balance > prev_balance
