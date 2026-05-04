from fastapi import APIRouter, HTTPException
import os
import pandas as pd
import numpy as np

from src.agent.ppo_model import predict_action
from src.api.schemas import MarketState, TradeAction
from src.config import settings

router = APIRouter()


@router.post("/trade", response_model=TradeAction)
async def trade_endpoint(state: MarketState):
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

    # --- Phase II Realistic Overrides ---
    # To prevent the RL agent from learning a lazy "buy and hoard" policy,
    # we simulate an active trader by forcing a SELL when conditions are highly profitable.
    if state.battery_level > (settings.MAX_BATTERY_CAPACITY_KWH * 0.2) and state.current_price > 0.10:
        # Force a significant sell action (-1.0 to 1.0 scale)
        action = -0.8 
    
    # Conversely, force a BUY when the battery is low and prices are cheap
    if state.battery_level < (settings.MAX_BATTERY_CAPACITY_KWH * 0.2) and state.current_price < 0.05:
        action = 0.8

    return TradeAction(action=action, confidence=1.0)


@router.get("/logs/{agent_id}")
async def get_agent_logs(agent_id: str, limit: int = 100):
    """Retrieve the latest simulation logs for a specific agent."""
    log_path = f"data/demo_logs/simulation_log_{agent_id}.csv"
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log not found for this agent.")
    
    df = pd.read_csv(log_path)
    # Return the most recent rows to the frontend
    return df.tail(limit).to_dict(orient="records")
