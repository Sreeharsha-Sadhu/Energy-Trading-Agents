from pydantic import BaseModel, Field


class MarketState(BaseModel):
    current_price: float = Field(..., ge=0)
    forecasted_demand: float = Field(..., ge=0)
    battery_level: float = Field(..., ge=0)
    account_balance: float = Field(...)


class TradeAction(BaseModel):
    action: int = Field(..., ge=0, le=2, description="0: Buy, 1: Sell, 2: Hold")
    confidence: float = Field(default=1.0)
