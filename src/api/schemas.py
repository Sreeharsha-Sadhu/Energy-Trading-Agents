from pydantic import BaseModel, Field


class MarketState(BaseModel):
    """Marketstate."""

    current_price: float = Field(..., ge=0)
    forecasted_demand: float = Field(..., ge=0)
    battery_level: float = Field(..., ge=0)
    account_balance: float = Field(...)


class TradeAction(BaseModel):
    """Tradeaction."""

    action: float = Field(..., ge=-1.0, le=1.0, description="-1.0: 100% Sell, 1.0: 100% Buy, 0.0: Hold")
    confidence: float = Field(default=1.0)
