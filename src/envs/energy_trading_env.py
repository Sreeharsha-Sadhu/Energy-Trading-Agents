import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.config import settings


class EnergyTradingEnv(gym.Env):
    """
    Custom Environment that follows gym interface for Energy Trading.
    """

    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode=None):
        super(EnergyTradingEnv, self).__init__()

        self.render_mode = render_mode

        # Action Space: 0 = Buy, 1 = Sell, 2 = Hold
        self.action_space = spaces.Discrete(3)

        # Observation Space: [Current_Price, Forecasted_Demand, Battery_Level, Account_Balance]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        # Initialize state variables
        self.current_price = 0.0
        self.forecasted_demand = 0.0
        self.battery_level = settings.INITIAL_BATTERY_KWH
        self.account_balance = settings.INITIAL_ACCOUNT_BALANCE

        self.current_step = 0
        self.max_steps = 24  # Example: 24 hours per episode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset state
        self.battery_level = settings.INITIAL_BATTERY_KWH
        self.account_balance = settings.INITIAL_ACCOUNT_BALANCE
        self.current_price = np.random.uniform(0.1, 0.5)  # random mock price
        self.forecasted_demand = np.random.uniform(1.0, 5.0)  # random mock demand
        self.current_step = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        return np.array(
            [
                self.current_price,
                self.forecasted_demand,
                self.battery_level,
                self.account_balance,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        reward = 0.0
        penalty_for_invalid_action = 0.0
        profit_from_trade = 0.0

        trade_volume = settings.MAX_TRADE_VOLUME_KWH
        price = self.current_price

        # Execute Action
        if action == 0:  # Buy
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
                penalty_for_invalid_action += 10.0  # Penalty
        elif action == 1:  # Sell
            revenue = trade_volume * price
            if self.battery_level >= trade_volume:
                self.battery_level -= trade_volume
                self.account_balance += revenue
                profit_from_trade = revenue
            else:
                penalty_for_invalid_action += 10.0  # Penalty
        elif action == 2:  # Hold
            pass

        # Apply external demand
        demand = self.forecasted_demand
        unmet_demand = 0.0
        if self.battery_level >= demand:
            self.battery_level -= demand
        else:
            unmet_demand = demand - self.battery_level
            self.battery_level = 0.0

        cost_of_unmet_demand = (
            unmet_demand * self.current_price * 2.0
        )  # Penalty multiplier

        # Calculate final reward
        reward = profit_from_trade - penalty_for_invalid_action - cost_of_unmet_demand

        # Move to next state
        self.current_step += 1
        self.current_price = np.random.uniform(0.1, 0.5)
        self.forecasted_demand = np.random.uniform(1.0, 5.0)

        terminated = self.current_step >= self.max_steps
        truncated = False

        if self.account_balance < 0:
            terminated = True

        obs = self._get_obs()
        info = {
            "battery_level": self.battery_level,
            "account_balance": self.account_balance,
            "unmet_demand": unmet_demand,
            "penalty": penalty_for_invalid_action,
            "profit": profit_from_trade,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "console":
            print(
                f"Step: {self.current_step}, Price: {self.current_price:.2f}, "
                f"Demand: {self.forecasted_demand:.2f}, Battery: {self.battery_level:.2f}, "
                f"Balance: {self.account_balance:.2f}"
            )
