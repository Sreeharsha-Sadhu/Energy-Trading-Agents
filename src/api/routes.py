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
            state.current_price / 0.20,
            state.forecasted_demand / 4.0,
            state.battery_level / settings.MAX_BATTERY_CAPACITY_KWH,
            state.account_balance / settings.INITIAL_ACCOUNT_BALANCE,
        ],
        dtype=np.float32,
    )

    action_int = predict_action(obs)

    return TradeAction(action=action_int, confidence=1.0)
