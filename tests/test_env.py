import numpy as np
import pytest

from src.config import settings
from src.envs.energy_trading_env import EnergyTradingEnv

# Continuous action helpers
BUY = np.array([1.0], dtype=np.float32)
SELL = np.array([-1.0], dtype=np.float32)
HOLD = np.array([0.0], dtype=np.float32)


@pytest.fixture
def env():
    e = EnergyTradingEnv()
    e.reset()
    return e


def test_env_reset_loads_episode_data(env):
    assert env.current_episode_data is not None
    assert len(env.current_episode_data) == env.max_steps
    # New schema: price, actual_demand, predicted_demand
    assert {"price", "actual_demand", "predicted_demand"}.issubset(
        env.current_episode_data.columns
    )
    obs, _ = env.reset()
    assert env.observation_space.shape == (4,)
    assert obs.dtype == np.float32


def test_sell_empty_battery():
    env = EnergyTradingEnv()
    env.reset()
    env.battery_level = 0.0
    _, reward, _, _, _ = env.step(SELL)
    assert reward < 0
    assert env.battery_level == 0.0


def test_buy_insufficient_balance():
    env = EnergyTradingEnv()
    env.reset()
    env.account_balance = 0.0
    _, reward, _, _, _ = env.step(BUY)
    assert reward < 0
    assert env.account_balance == 0.0


def test_buy_full_battery():
    env = EnergyTradingEnv()
    env.reset()
    env.battery_level = settings.MAX_BATTERY_CAPACITY_KWH
    env.account_balance = 1000.0
    # Observation uses predicted_demand, but actual_demand is read from episode data.
    # Since battery is full, BUY action should fail.
    initial_battery = env.battery_level
    _, reward, _, _, _ = env.step(BUY)
    assert reward < 0
    # Battery should not increase (full), and may decrease due to actual_demand
    assert env.battery_level <= initial_battery


def test_valid_buy_and_sell():
    env = EnergyTradingEnv()
    env.reset()
    env.battery_level = settings.MAX_BATTERY_CAPACITY_KWH / 2
    env.account_balance = 100.0
    env.current_price = 1.0

    _, _, _, _, _ = env.step(BUY)
    assert env.battery_level > settings.MAX_BATTERY_CAPACITY_KWH / 2
    assert env.account_balance < 100.0

    env.current_price = 2.0
    prev_balance = env.account_balance
    prev_battery = env.battery_level
    _, _, _, _, _ = env.step(SELL)
    assert env.battery_level < prev_battery
    assert env.account_balance > prev_balance


def test_variance_penalty_in_info():
    env = EnergyTradingEnv()
    env.reset()
    _, _, _, _, info = env.step(HOLD)
    assert "variance_penalty" in info
    assert info["variance_penalty"] >= 0.0


def test_variance_penalty_non_zero_after_enough_samples():
    """After _VARIANCE_MIN_SAMPLES + 1 steps with non-zero profit, the penalty
    must be positive (std of non-constant profits > 0)."""
    env = EnergyTradingEnv()
    env.reset()

    # Alternate BUY/SELL to generate profit fluctuations.
    env.account_balance = 10_000.0
    env.current_price = 0.10
    last_penalty = 0.0
    for i in range(env._VARIANCE_MIN_SAMPLES + 2):
        action = BUY if i % 2 == 0 else SELL
        env.battery_level = settings.MAX_BATTERY_CAPACITY_KWH / 2
        _, _, _, _, info = env.step(action)
        last_penalty = info["variance_penalty"]

    assert last_penalty > 0.0, (
        "Expected a positive variance penalty after enough profit fluctuations"
    )
