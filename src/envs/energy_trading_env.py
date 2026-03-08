import gymnasium as gym
from gymnasium import spaces
import numpy as np
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
        super(EnergyTradingEnv, self).__init__()

        self.render_mode = render_mode

        # Action Space: 0 = Buy, 1 = Sell, 2 = Hold
        self.action_space = spaces.Discrete(3)

        # Observation Space: [Current_Price, Forecasted_Demand, Battery_Level, Account_Balance]
        # Normalized to roughly [0, 1] range for stable training
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(4,), dtype=np.float32
        )

        # State variables
        self.current_price = 0.0
        self.forecasted_demand = 0.0
        self.battery_level = settings.INITIAL_BATTERY_KWH
        self.account_balance = settings.INITIAL_ACCOUNT_BALANCE

        # Track the running average price for buy-low/sell-high incentive
        self._price_history: list[float] = []

        self.current_step = 0
        self.max_steps = 48  # 48 hours per episode (2 days)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.battery_level = settings.INITIAL_BATTERY_KWH
        self.account_balance = settings.INITIAL_ACCOUNT_BALANCE
        self.current_price = np.random.uniform(0.05, 0.20)
        self.forecasted_demand = np.random.uniform(0.5, 4.0)
        self._price_history = [self.current_price]
        self.current_step = 0

        return self._get_obs(), {}

    def _get_obs(self):
        # Normalize observations for stable neural network training
        return np.array(
            [
                self.current_price / 0.20,                           # ~[0.25, 5.0]
                self.forecasted_demand / 4.0,                        # ~[0.125, 1.0]
                self.battery_level / settings.MAX_BATTERY_CAPACITY_KWH,  # [0, 1]
                self.account_balance / settings.INITIAL_ACCOUNT_BALANCE, # ~[0, 2]
            ],
            dtype=np.float32,
        )

    def _avg_price(self) -> float:
        if not self._price_history:
            return self.current_price
        return float(np.mean(self._price_history[-24:]))  # 24-hour rolling average

    def step(self, action):
        reward = 0.0
        trade_volume = settings.MAX_TRADE_VOLUME_KWH
        price = self.current_price
        avg_price = self._avg_price()
        profit_from_trade = 0.0
        penalty = 0.0

        # Execute action
        if action == 0:  # Buy
            cost = trade_volume * price
            if (
                self.account_balance >= cost
                and self.battery_level + trade_volume <= settings.MAX_BATTERY_CAPACITY_KWH
            ):
                self.account_balance -= cost
                self.battery_level += trade_volume
                profit_from_trade = -cost
                # Bonus for buying below average price (buy low)
                if price < avg_price:
                    reward += (avg_price - price) * trade_volume * 5.0
                else:
                    # Small penalty for buying at above-average price
                    reward -= (price - avg_price) * trade_volume * 2.0
            else:
                penalty = 5.0  # Penalty for invalid buy

        elif action == 1:  # Sell
            revenue = trade_volume * price
            if self.battery_level >= trade_volume:
                self.battery_level -= trade_volume
                self.account_balance += revenue
                profit_from_trade = revenue
                # Bonus for selling above average price (sell high)
                if price > avg_price:
                    reward += (price - avg_price) * trade_volume * 5.0
                else:
                    # Small penalty for selling below average
                    reward -= (avg_price - price) * trade_volume * 2.0
            else:
                penalty = 5.0  # Penalty for invalid sell

        elif action == 2:  # Hold
            # Small reward for holding when price is near average (waiting for opportunity)
            reward += 0.01

        # Apply external demand drain
        demand = self.forecasted_demand
        unmet_demand = 0.0
        if self.battery_level >= demand:
            self.battery_level -= demand
        else:
            unmet_demand = demand - self.battery_level
            self.battery_level = 0.0

        # Penalty for unmet demand (proportional but not overwhelming)
        unmet_penalty = unmet_demand * price * 3.0

        # Battery level health bonus: reward for maintaining reasonable reserves
        battery_ratio = self.battery_level / settings.MAX_BATTERY_CAPACITY_KWH
        if 0.2 <= battery_ratio <= 0.8:
            reward += 0.05  # Small bonus for healthy battery range
        elif battery_ratio < 0.1:
            reward -= 0.1  # Penalty for dangerously low battery

        # Final reward
        reward = reward + profit_from_trade - penalty - unmet_penalty

        # Advance to next state
        self.current_step += 1

        # Generate next market conditions with time-varying patterns
        hour_of_day = self.current_step % 24
        # Price follows a sinusoidal pattern: higher during peak hours (8-20)
        base_price = 0.10 + 0.05 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        self.current_price = float(np.clip(base_price + np.random.normal(0, 0.03), 0.02, 0.40))

        # Demand follows its own cycle: higher during day, lower at night
        base_demand = 2.0 + 1.5 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)
        self.forecasted_demand = float(np.clip(base_demand + np.random.normal(0, 0.5), 0.2, 5.0))

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
        if self.render_mode == "console":
            print(
                f"Step: {self.current_step}, Price: {self.current_price:.3f}, "
                f"Demand: {self.forecasted_demand:.2f}, Battery: {self.battery_level:.2f}, "
                f"Balance: {self.account_balance:.2f}"
            )
