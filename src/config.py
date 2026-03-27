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

    DEMO_FORECAST_SEGMENT: str = "Residential_Solar"
    DEMO_FORECAST_MIN_DEMAND_KWH: float = 0.1
    DEMO_FORECAST_HISTORY_HOURS: int = 168
    DEMO_FALLBACK_NOISE_STD: float = 0.08


settings = Settings()
