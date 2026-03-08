import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings."""

    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))

    MAX_BATTERY_CAPACITY_KWH: float = 50.0
    INITIAL_BATTERY_KWH: float = 10.0
    MAX_TRADE_VOLUME_KWH: float = 5.0
    INITIAL_ACCOUNT_BALANCE: float = 100.0

    MODEL_SAVE_PATH: str = "models/ppo_energy_agent.zip"
    TOTAL_TIMESTEPS: int = 100000


settings = Settings()
