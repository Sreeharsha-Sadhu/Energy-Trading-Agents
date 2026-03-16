"""Train the PPO energy trading agent used by the live demo.

Usage:
    PYTHONPATH=. uv run scripts/train_demo_agent.py --timesteps 200000
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.envs.energy_trading_env import EnergyTradingEnv


def main():
    """Main."""
    parser = argparse.ArgumentParser(description="Train the PPO demo agent")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/ppo_energy_agent",
        help="Path to save the trained model (without .zip extension)",
    )
    args = parser.parse_args()

    env = make_vec_env(EnergyTradingEnv, n_envs=4)

    print(f"🏋️  Training PPO agent for {args.timesteps:,} timesteps …")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    model.learn(total_timesteps=args.timesteps)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"✅ Model saved to {args.output}.zip")


if __name__ == "__main__":
    main()
