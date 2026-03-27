"""Simulation orchestrator for the Energy Trading Agent demo.

Replays synthetic market data, sends each tick to the FastAPI /api/v1/trade
endpoint, applies the returned action locally (mirroring the gym env logic),
and logs every state transition to a CSV for the dashboard to consume.

Supports dynamic scenario overrides via a JSON control file written by the
Streamlit dashboard sidebar.
"""

import argparse
import collections
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from src.config import settings
from src.demo.data_provider import iterate_market_ticks
from src.forecaster.modeling.inference import RealtimeDemandForecaster

SCENARIO_CONTROL_FILE = "data/demo_logs/scenario_control.json"


def _read_scenario_overrides() -> dict:
    """Read dynamic scenario overrides written by the dashboard."""
    defaults = {"price_multiplier": 1.0, "demand_multiplier": 1.0}
    if not os.path.exists(SCENARIO_CONTROL_FILE):
        return defaults
    try:
        with open(SCENARIO_CONTROL_FILE, "r") as f:
            data = json.load(f)
        return {
            "price_multiplier": float(data.get("price_multiplier", 1.0)),
            "demand_multiplier": float(data.get("demand_multiplier", 1.0)),
            "scenario_name": str(data.get("scenario_name", "")),
        }
    except Exception:
        return defaults


def run_simulation(
    speed: float = 1.0,
    hours: int = 168,
    log_dir: str = "data/demo_logs",
    start_date: datetime | None = None,
):
    """Run the trading simulation loop.

    Args:
        speed: Simulation speed multiplier (1 real sec = X sim hours).
        hours: Total number of simulation hours to run.
        log_dir: Directory for the CSV log consumed by the dashboard.

    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "simulation_log.csv")

    if start_date is None:
        start_date = datetime.now()

    balance = settings.INITIAL_ACCOUNT_BALANCE
    battery_level = settings.INITIAL_BATTERY_KWH
    trade_volume = settings.MAX_TRADE_VOLUME_KWH
    max_battery = settings.MAX_BATTERY_CAPACITY_KWH
    initial_balance = balance

    # Mirror env constants so logged values match what PPO trained on.
    _VARIANCE_WINDOW = 24
    _VARIANCE_MIN_SAMPLES = 5
    _VARIANCE_PENALTY_SCALE = 0.05
    profit_history: collections.deque[float] = collections.deque(
        maxlen=_VARIANCE_WINDOW
    )
    last_logged_scenario = ""
    history: list[dict] = []
    demand_history: collections.deque[float] = collections.deque(
        maxlen=settings.DEMO_FORECAST_HISTORY_HOURS * 2
    )
    forecaster = RealtimeDemandForecaster(segment_name=settings.DEMO_FORECAST_SEGMENT)

    print(f"🚀 Starting simulation: {hours} hours at {speed}x speed")
    print(f"   Initial balance: ${balance:.2f} | Battery: {battery_level:.1f} kWh")
    print(
        f"   Trade volume: {trade_volume:.1f} kWh | Max battery: {max_battery:.1f} kWh"
    )
    print(f"📊 Logging to: {log_file}")

    for dt, base_price, base_demand in iterate_market_ticks(
        start_date, num_hours=hours
    ):
        overrides = _read_scenario_overrides()
        price = base_price * overrides["price_multiplier"]
        actual_demand = base_demand * overrides["demand_multiplier"]
        predicted_demand = forecaster.predict_demand(
            dt=dt,
            actual_demand=float(actual_demand),
            demand_history=demand_history,
        )

        action_name = "HOLD"

        state = {
            "current_price": float(price),
            "forecasted_demand": float(predicted_demand),
            "battery_level": float(battery_level),
            "account_balance": float(balance),
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/v1/trade",
                json=state,
                timeout=2,
            )
            if response.status_code == 200:
                data = response.json()
                action_val = float(data.get("action", 0.0))

                # Proportional trade volume mirroring environment logic
                step_trade_volume = abs(action_val) * settings.MAX_TRADE_VOLUME_KWH

                if action_val > 0.05:
                    action_name = "BUY"
                elif action_val < -0.05:
                    action_name = "SELL"
                else:
                    action_name = "HOLD"
            else:
                action_name = "HOLD"
                step_trade_volume = 0.0
        except Exception:
            action_name = "HOLD"
            step_trade_volume = 0.0

        cost = step_trade_volume * price
        revenue = step_trade_volume * price
        step_profit = 0.0

        if action_name == "BUY":
            if balance >= cost and battery_level + step_trade_volume <= max_battery:
                battery_level += step_trade_volume
                balance -= cost
                step_profit = -cost
        elif action_name == "SELL":
            if battery_level >= step_trade_volume:
                battery_level -= step_trade_volume
                balance += revenue
                step_profit = revenue

        unmet_demand = 0.0
        if battery_level >= actual_demand:
            battery_level -= actual_demand
        else:
            unmet_demand = actual_demand - battery_level
            battery_level = 0.0

        demand_history.append(float(actual_demand))

        # Compute variance penalty matching the env reward shaping.
        profit_history.append(step_profit)
        variance_penalty = 0.0
        if len(profit_history) > _VARIANCE_MIN_SAMPLES:
            variance_penalty = float(np.std(profit_history)) * _VARIANCE_PENALTY_SCALE

        # Reset scenario log once written to avoid duplicate lines on same scenario
        current_scenario = overrides.get("scenario_name", "")
        scenario_to_log = ""
        if current_scenario != last_logged_scenario:
            scenario_to_log = current_scenario
        last_logged_scenario = current_scenario

        log_entry = {
            "sim_datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "price": round(price, 4),
            "demand": round(actual_demand, 4),
            "actual_demand": round(actual_demand, 4),
            "predicted_demand": round(predicted_demand, 4),
            "forecast_error": round(actual_demand - predicted_demand, 4),
            "action_name": action_name,
            "battery_level": round(battery_level, 2),
            "account_balance": round(balance, 2),
            "cumulative_profit": round(balance - initial_balance, 2),
            "unmet_demand": round(unmet_demand, 2),
            "reward": 0.0,
            "variance_penalty": round(variance_penalty, 6),
            "price_multiplier": overrides["price_multiplier"],
            "demand_multiplier": overrides["demand_multiplier"],
            "active_scenario": scenario_to_log,
        }
        history.append(log_entry)

        pd.DataFrame(history).to_csv(log_file, index=False)

        time.sleep(1.0 / speed)

        if len(history) % 24 == 0:
            print(
                f"✅ {dt}: Balance=${balance:.2f}, "
                f"Battery={battery_level:.1f} kWh, "
                f"P&L=${(balance - initial_balance):.2f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Energy Trading Agent simulation runner"
    )
    parser.add_argument(
        "--speed", type=float, default=1, help="Sim speed: 1 real sec = X sim hours"
    )
    parser.add_argument("--hours", type=int, default=168, help="Total sim hours")
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Simulation start datetime in ISO format (e.g., 2025-03-27 or 2025-03-27T00:00:00)",
    )
    args = parser.parse_args()

    parsed_start_date: datetime | None = None
    if args.start_date:
        parsed_start_date = datetime.fromisoformat(args.start_date)

    run_simulation(speed=args.speed, hours=args.hours, start_date=parsed_start_date)
