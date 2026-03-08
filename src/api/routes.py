import numpy as np
from fastapi import APIRouter
from src.api.schemas import MarketState, TradeAction
from src.agent.ppo_model import predict_action
from src.config import settings

router = APIRouter()


@router.post("/trade", response_model=TradeAction)
async def trade_endpoint(state: MarketState):
    # Convert payload to normalized NumPy array matching the training env's
    # _get_obs() normalization: price/0.20, demand/4.0, battery/50.0, balance/100.0
    obs = np.array(
        [
            state.current_price / 0.20,
            state.forecasted_demand / 4.0,
            state.battery_level / settings.MAX_BATTERY_CAPACITY_KWH,
            state.account_balance / settings.INITIAL_ACCOUNT_BALANCE,
        ],
        dtype=np.float32,
    )

    # Get action from the agent
    action_int = predict_action(obs)

    return TradeAction(action=action_int, confidence=1.0)
