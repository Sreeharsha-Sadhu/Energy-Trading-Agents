import numpy as np
from fastapi import APIRouter
from src.api.schemas import MarketState, TradeAction
from src.agent.ppo_model import predict_action

router = APIRouter()


@router.post("/trade", response_model=TradeAction)
async def trade_endpoint(state: MarketState):
    # Convert payload to NumPy array matching observation space
    obs = np.array(
        [
            state.current_price,
            state.forecasted_demand,
            state.battery_level,
            state.account_balance,
        ],
        dtype=np.float32,
    )

    # Get action from the agent
    action_int = predict_action(obs)

    return TradeAction(action=action_int, confidence=1.0)
