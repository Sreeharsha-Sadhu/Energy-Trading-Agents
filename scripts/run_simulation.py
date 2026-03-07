#!/usr/bin/env python3
import time
import requests
import json
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta
from src.demo.data_provider import iterate_market_ticks

def run_simulation(speed=1, hours=168, log_dir="data/demo_logs"):
    """
    speed: simulation speed (1 real second = X simulation hours)
    hours: total simulation hours
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "simulation_log.csv")
    
    # Initialize components
    start_date = datetime.now()
    
    # Initial state
    balance = 10000.0
    battery_level = 0.0
    history = []
    
    print(f"🚀 Starting simulation: {hours} hours at {speed}x speed")
    print(f"📊 Logging to: {log_file}")
    
    # Simulation loop using the data provider generator
    for dt, price, demand in iterate_market_ticks(start_date, num_hours=hours):
        action_name = "HOLD"
        confidence = 1.0
        
        # Prepare state for API – MUST match src/api/schemas.py
        state = {
            "current_price": float(price),
            "forecasted_demand": float(demand),
            "battery_level": float(battery_level),
            "account_balance": float(balance)
        }
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/v1/trade",
                json=state,
                timeout=2
            )
            if response.status_code == 200:
                data = response.json()
                action_idx = data.get("action", 2)
                # API Schema: 0: Buy, 1: Sell, 2: Hold
                action_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
                action_name = action_map.get(action_idx, "HOLD")
                confidence = data.get("confidence", 1.0)
            else:
                print(f"⚠️ API Error at {dt}: {response.text}")
                action_name = "HOLD"
        except Exception as e:
            # print(f"⚠️ App Exception at {dt}: {e}")
            action_name = "HOLD"
            
        # 4. Perform Action & Update Financials (Local simulation of trade outcome)
        if action_name == "BUY" and balance >= price:
            if battery_level < 50.0:
                battery_level += 1.0
                balance -= price
        elif action_name == "SELL" and battery_level >= 1.0:
            battery_level -= 1.0
            balance += price
            
        # 5. Log entry for Dashboard
        log_entry = {
            "sim_datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "price": price,
            "demand": demand,
            "action_name": action_name,
            "battery_level": battery_level,
            "account_balance": balance,
            "cumulative_profit": balance - 10000.0,
            "unmet_demand": 0.0,
            "reward": 0.0
        }
        history.append(log_entry)
        
        # 6. Update Live Log File
        pd.DataFrame(history).to_csv(log_file, index=False)
            
        # 7. Sleep for effect
        time.sleep(1.0 / speed)
        
        if len(history) % 24 == 0:
            print(f"✅ {dt}: Balance=${balance:.2f}, P&L=${(balance-10000.0):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1, help="Sim speed: 1 real sec = X sim hours")
    parser.add_argument("--hours", type=int, default=168, help="Total sim hours")
    args = parser.parse_args()
    
    run_simulation(speed=args.speed, hours=args.hours)
