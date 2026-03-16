import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import settings


class EnergyTradingEnv(gym.Env):
    """Custom Gymnasium environment for energy trading with battery storage.

    The agent manages a battery storage system in a simulated energy market.
    It observes the current market price, forecasted demand, battery level,
    and account balance, then decides to Buy, Sell, or Hold energy.

    The reward function incentivizes profitable trading (buy low, sell high)
    while penalizing unmet demand and invalid actions.
    """

    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode=None):
        """Init."""
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

        self.current_step = 0
        self.max_steps = 48  # 48 hours per episode (2 days)

    def reset(self, seed=None, options=None):
        """Reset."""
        super().reset(seed=seed)

        self.battery_level = settings.INITIAL_BATTERY_KWH
        self.account_balance = settings.INITIAL_ACCOUNT_BALANCE
        self.current_price = np.random.uniform(0.05, 0.20)
        self.forecasted_demand = np.random.uniform(0.5, 4.0)
        self._price_history = [self.current_price]
        self.current_step = 0

        return self._get_obs(), {}

    def _get_obs(self):
        """Get Obs."""
        return np.array(
            [
                np.clip(self.current_price / 0.40, 0.0, 1.0),
                np.clip(self.forecasted_demand / 5.0, 0.0, 1.0),
                np.clip(self.battery_level / settings.MAX_BATTERY_CAPACITY_KWH, 0.0, 1.0),
                np.clip(self.account_balance / (settings.INITIAL_ACCOUNT_BALANCE * 2), 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _avg_price(self) -> float:
        """Avg Price."""
        if not self._price_history:
            return self.current_price
        return float(np.mean(self._price_history[-24:]))  # 24-hour rolling average

    def step(self, action):
        """Step."""
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
                and self.battery_level + trade_volume <= settings.MAX_BATTERY_CAPACITY_KWH
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

        demand = self.forecasted_demand
        unmet_demand = 0.0
        if self.battery_level >= demand:
            self.battery_level -= demand
        else:
            unmet_demand = demand - self.battery_level
            self.battery_level = 0.0

        unmet_penalty = unmet_demand * price * 3.0

        profit_reward = profit_from_trade * REWARD_SCALE
        unmet_penalty_scaled = unmet_penalty * REWARD_SCALE
        reward = profit_reward - penalty - unmet_penalty_scaled

        self.current_step += 1

        hour_of_day = self.current_step % 24
        base_price = 0.10 + 0.05 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        self.current_price = float(
            np.clip(base_price + np.random.normal(0, 0.03), 0.02, 0.40)
        )

        base_demand = 2.0 + 1.5 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)
        self.forecasted_demand = float(
            np.clip(base_demand + np.random.normal(0, 0.5), 0.2, 5.0)
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
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Render."""
        if self.render_mode == "console":
            print(
                f"Step: {self.current_step}, Price: {self.current_price:.3f}, "
                f"Demand: {self.forecasted_demand:.2f}, Battery: {self.battery_level:.2f}, "
                f"Balance: {self.account_balance:.2f}"
            )
