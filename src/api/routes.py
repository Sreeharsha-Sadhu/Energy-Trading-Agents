import numpy as np
from fastapi import APIRouter

from src.agent.ppo_model import predict_action
from src.api.schemas import MarketState, TradeAction
from src.config import settings

router = APIRouter()


@router.post("/trade", response_model=TradeAction)
async def trade_endpoint(state: MarketState):
    """Trade Endpoint."""
    obs = np.array(
        [
            np.clip(state.current_price / 0.40, 0.0, 1.0),
            np.clip(state.forecasted_demand / 5.0, 0.0, 1.0),
            np.clip(state.battery_level / settings.MAX_BATTERY_CAPACITY_KWH, 0.0, 1.0),
            np.clip(state.account_balance / (settings.INITIAL_ACCOUNT_BALANCE * 2), 0.0, 1.0),
        ],
        dtype=np.float32,
    )

    action = predict_action(obs)

    return TradeAction(action=action, confidence=1.0)
