import collections
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import settings
from src.forecaster.modeling.inference import get_forecast_data


class MultiAgentEnergyTradingEnv(gym.Env):
    """
    Multi-Agent Gymnasium environment for energy trading.
    This simulates multiple actors (e.g., a Residential battery agent and an Industrial agent)
    interacting in the same shared market.
    """

    metadata = {"render_modes": ["console"]}

    def __init__(self, num_agents=2, render_mode=None):
        super().__init__()
        self.num_agents = num_agents
        self.render_mode = render_mode

        # Multi-agent action and observation spaces
        # Action: Buy/Sell/Hold for each agent
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_agents,), dtype=np.float32)
        
        # Observation: Shared market price, demand, and individual battery & balance for each agent
        # Size = 2 (shared: price, demand) + 2 * num_agents (battery, balance)
        obs_size = 2 + (2 * num_agents)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        self.current_price = 0.0
        self.forecasted_demand = 0.0
        
        self.battery_levels = [settings.INITIAL_BATTERY_KWH for _ in range(num_agents)]
        self.account_balances = [settings.INITIAL_ACCOUNT_BALANCE for _ in range(num_agents)]
        
        self.current_episode_data = None
        self.current_step = 0
        self.max_steps = 48

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.battery_levels = [settings.INITIAL_BATTERY_KWH for _ in range(self.num_agents)]
        self.account_balances = [settings.INITIAL_ACCOUNT_BALANCE for _ in range(self.num_agents)]
        self.current_step = 0

        self.current_episode_data = get_forecast_data(window_size=self.max_steps)
        self.current_price = float(self.current_episode_data.iloc[0]["price"])
        self.forecasted_demand = float(self.current_episode_data.iloc[0]["predicted_demand"])

        return self._get_obs(), {}

    def _get_obs(self):
        obs = [
            np.clip(self.current_price / 0.40, 0.0, 1.0),
            np.clip(self.forecasted_demand / 5.0, 0.0, 1.0),
        ]
        for i in range(self.num_agents):
            obs.append(np.clip(self.battery_levels[i] / settings.MAX_BATTERY_CAPACITY_KWH, 0.0, 1.0))
            obs.append(np.clip(self.account_balances[i] / (settings.INITIAL_ACCOUNT_BALANCE * 2), 0.0, 1.0))
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        REWARD_SCALE = 0.01
        
        total_market_action = sum(actions)
        
        # Apply market price impact from aggregated actions
        price = self.current_price
        if getattr(settings, "ENABLE_PRICE_IMPACT", False):
            slippage_factor = 0.05
            price = price * (1 + (total_market_action * slippage_factor))

        rewards = np.zeros(self.num_agents, dtype=np.float32)
        
        actual_demand = float(self.current_episode_data.iloc[self.current_step]["actual_demand"])
        
        for i in range(self.num_agents):
            action_val = float(np.clip(actions[i], -1.0, 1.0))
            trade_volume = abs(action_val) * settings.MAX_TRADE_VOLUME_KWH
            
            profit_from_trade = 0.0
            penalty = 0.0
            
            if action_val > 0.05:  # Buy
                cost = trade_volume * price
                if self.account_balances[i] >= cost and self.battery_levels[i] + trade_volume <= settings.MAX_BATTERY_CAPACITY_KWH:
                    self.account_balances[i] -= cost
                    self.battery_levels[i] += trade_volume
                    profit_from_trade = -cost
                else:
                    penalty = 1.0
            elif action_val < -0.05:  # Sell
                revenue = trade_volume * price
                if self.battery_levels[i] >= trade_volume:
                    self.battery_levels[i] -= trade_volume
                    self.account_balances[i] += revenue
                    profit_from_trade = revenue
                else:
                    penalty = 1.0
            
            # Simplified per-agent demand
            agent_actual_demand = actual_demand / self.num_agents
            unmet_demand = 0.0
            if self.battery_levels[i] >= agent_actual_demand:
                self.battery_levels[i] -= agent_actual_demand
            else:
                unmet_demand = agent_actual_demand - self.battery_levels[i]
                self.battery_levels[i] = 0.0
                
            unmet_penalty = unmet_demand * price * 3.0
            
            imbalance_penalty = 0.0
            if getattr(settings, "ENABLE_IMBALANCE_PENALTY", False):
                agent_forecasted_demand = self.forecasted_demand / self.num_agents
                deviation = abs(agent_actual_demand - agent_forecasted_demand)
                imbalance_penalty = deviation * price * 2.0
                
            rewards[i] = (profit_from_trade - penalty - unmet_penalty - imbalance_penalty) * REWARD_SCALE

        self.current_step += 1

        if self.current_step < self.max_steps:
            self.current_price = float(self.current_episode_data.iloc[self.current_step]["price"])
            self.forecasted_demand = float(self.current_episode_data.iloc[self.current_step]["predicted_demand"])

        terminated = self.current_step >= self.max_steps
        truncated = False

        if any(b < 0 for b in self.account_balances):
            terminated = True

        # Use average reward for single-agent compatibility if wrapped, or return array depending on framework.
        # Here we sum the rewards as a simple cooperative scalar reward since the space is standard gym.Env.
        # For true independent MARL, a PettingZoo wrapper should be used around this.
        scalar_reward = float(np.sum(rewards))

        info = {
            "battery_levels": self.battery_levels.copy(),
            "account_balances": self.account_balances.copy(),
        }

        return self._get_obs(), scalar_reward, terminated, truncated, info
