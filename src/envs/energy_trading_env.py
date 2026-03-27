import collections

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import settings
from src.forecaster.modeling.inference import get_forecast_data


class EnergyTradingEnv(gym.Env):
    """Custom Gymnasium environment for energy trading with battery storage.

    The agent manages a battery storage system in a simulated energy market.
    It observes the current market price, forecasted demand, battery level,
    and account balance, then decides to Buy, Sell, or Hold energy.

    The reward function incentivizes profitable trading (buy low, sell high)
    while penalizing unmet demand and invalid actions.
    """

    metadata = {"render_modes": ["console"]}

    # Number of recent profit steps to track for variance penalty.
    _VARIANCE_WINDOW = 24
    # Minimum samples before the variance penalty is applied.
    _VARIANCE_MIN_SAMPLES = 5
    # Scaling factor for the rolling-std risk penalty.
    _VARIANCE_PENALTY_SCALE = 0.05

    def __init__(self, render_mode=None):
        super(EnergyTradingEnv, self).__init__()

        self.render_mode = render_mode

        # Continuous action: -1.0 = 100% sell, +1.0 = 100% buy, ~0.0 = hold
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.current_price = 0.0
        self.forecasted_demand = 0.0
        self.battery_level = settings.INITIAL_BATTERY_KWH
        self.account_balance = settings.INITIAL_ACCOUNT_BALANCE

        self._price_history: list[float] = []
        # Episode data buffer populated on reset via get_forecast_data.
        self.current_episode_data = None
        # Rolling profit buffer used to compute the variance penalty.
        self.profit_history: collections.deque[float] = collections.deque(
            maxlen=self._VARIANCE_WINDOW
        )

        self.current_step = 0
        self.max_steps = 48

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.battery_level = settings.INITIAL_BATTERY_KWH
        self.account_balance = settings.INITIAL_ACCOUNT_BALANCE
        self.profit_history.clear()
        self.current_step = 0

        # Load a fresh window of market data for this episode.
        # get_forecast_data now returns price, actual_demand, predicted_demand.
        self.current_episode_data = get_forecast_data(window_size=self.max_steps)
        self.current_price = float(self.current_episode_data.iloc[0]["price"])
        # Agent observes predicted demand (imperfect foresight)
        self.forecasted_demand = float(
            self.current_episode_data.iloc[0]["predicted_demand"]
        )
        self._price_history = [self.current_price]

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(
            [
                np.clip(self.current_price / 0.40, 0.0, 1.0),
                np.clip(self.forecasted_demand / 5.0, 0.0, 1.0),
                np.clip(
                    self.battery_level / settings.MAX_BATTERY_CAPACITY_KWH, 0.0, 1.0
                ),
                np.clip(
                    self.account_balance / (settings.INITIAL_ACCOUNT_BALANCE * 2),
                    0.0,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )

    def _avg_price(self) -> float:
        if not self._price_history:
            return self.current_price
        return float(np.mean(self._price_history[-24:]))

    def step(self, action):
        REWARD_SCALE = 0.01

        action_val = float(np.clip(action[0], -1.0, 1.0))
        trade_volume = abs(action_val) * settings.MAX_TRADE_VOLUME_KWH
        price = self.current_price
        profit_from_trade = 0.0
        penalty = 0.0

        if action_val > 0.05:  # Buy
            cost = trade_volume * price
            if (
                self.account_balance >= cost
                and self.battery_level + trade_volume
                <= settings.MAX_BATTERY_CAPACITY_KWH
            ):
                self.account_balance -= cost
                self.battery_level += trade_volume
                profit_from_trade = -cost
            else:
                penalty = 1.0  # Normalized penalty

        elif action_val < -0.05:  # Sell
            revenue = trade_volume * price
            if self.battery_level >= trade_volume:
                self.battery_level -= trade_volume
                self.account_balance += revenue
                profit_from_trade = revenue
            else:
                penalty = 1.0  # Normalized penalty

        # Battery drain uses actual_demand (ground truth consumption)
        actual_demand = float(
            self.current_episode_data.iloc[self.current_step]["actual_demand"]
        )
        unmet_demand = 0.0
        if self.battery_level >= actual_demand:
            self.battery_level -= actual_demand
        else:
            unmet_demand = actual_demand - self.battery_level
            self.battery_level = 0.0

        unmet_penalty = unmet_demand * price * 3.0

        # --- Risk-adjusted reward: rolling profit variance penalty ---
        self.profit_history.append(profit_from_trade)
        variance_penalty = 0.0
        if len(self.profit_history) > self._VARIANCE_MIN_SAMPLES:
            profit_std = float(np.std(self.profit_history))
            variance_penalty = profit_std * self._VARIANCE_PENALTY_SCALE

        profit_reward = profit_from_trade * REWARD_SCALE
        unmet_penalty_scaled = unmet_penalty * REWARD_SCALE
        reward = profit_reward - penalty - unmet_penalty_scaled - variance_penalty

        self.current_step += 1

        # Advance market data from episode buffer; stay on the last row if we
        # have somehow overrun (the terminated flag handles episode end).
        if self.current_step < self.max_steps:
            self.current_price = float(
                self.current_episode_data.iloc[self.current_step]["price"]
            )
            # Agent observes predicted_demand (imperfect foresight)
            self.forecasted_demand = float(
                self.current_episode_data.iloc[self.current_step]["predicted_demand"]
            )

        self._price_history.append(self.current_price)

        terminated = self.current_step >= self.max_steps
        truncated = False

        if self.account_balance < 0:
            terminated = True

        obs = self._get_obs()
        info = {
            "battery_level": self.battery_level,
            "account_balance": self.account_balance,
            "unmet_demand": unmet_demand,
            "penalty": penalty,
            "profit": profit_from_trade,
            "variance_penalty": variance_penalty,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "console":
            print(
                f"Step: {self.current_step}, Price: {self.current_price:.3f}, "
                f"Demand: {self.forecasted_demand:.2f}, Battery: {self.battery_level:.2f}, "
                f"Balance: {self.account_balance:.2f}"
            )
