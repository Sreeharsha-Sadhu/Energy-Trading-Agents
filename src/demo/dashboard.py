#!/usr/bin/env python3
"""
Streamlit live dashboard for the Energy Trading Agent demo.

Reads the simulation log CSV produced by run_simulation.py and renders
live KPI cards and Plotly time-series charts that auto-refresh.

Includes a sidebar for injecting real-time scenario overrides (price and
demand multipliers) that the background simulation script picks up dynamically.

Usage:
    streamlit run src/demo/dashboard.py
"""

import json
import os
import time

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="⚡ Energy Trading Agent — Live Demo",
    page_icon="⚡",
    layout="wide",
)

LOG_FILE = "data/demo_logs/simulation_log.csv"
SCENARIO_CONTROL_FILE = "data/demo_logs/scenario_control.json"
REFRESH_INTERVAL = 2  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=REFRESH_INTERVAL)
def load_log() -> pd.DataFrame:
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty:
            return df
        df["sim_datetime"] = pd.to_datetime(df["sim_datetime"])
        return df
    except Exception:
        return pd.DataFrame()


def write_scenario_overrides(price_mult: float, demand_mult: float):
    """Write scenario overrides to the shared JSON control file."""
    os.makedirs(os.path.dirname(SCENARIO_CONTROL_FILE), exist_ok=True)
    with open(SCENARIO_CONTROL_FILE, "w") as f:
        json.dump(
            {"price_multiplier": price_mult, "demand_multiplier": demand_mult},
            f,
        )


ACTION_COLORS = {"BUY": "#00c853", "SELL": "#ff1744", "HOLD": "#ffab00"}
ACTION_SYMBOLS = {"BUY": "triangle-up", "SELL": "triangle-down", "HOLD": "circle"}

# Preset scenario configurations
SCENARIO_PRESETS = {
    "🌤️ Normal Market": {"price_multiplier": 1.0, "demand_multiplier": 1.0},
    "📈 Price Spike": {"price_multiplier": 2.5, "demand_multiplier": 1.0},
    "📉 Price Crash": {"price_multiplier": 0.3, "demand_multiplier": 1.0},
    "🔥 Demand Surge": {"price_multiplier": 1.0, "demand_multiplier": 2.5},
    "❄️ Low Demand": {"price_multiplier": 1.0, "demand_multiplier": 0.4},
    "⚡ Crisis: High Price + High Demand": {"price_multiplier": 2.5, "demand_multiplier": 2.5},
    "💰 Opportunity: Low Price + Low Demand": {"price_multiplier": 0.3, "demand_multiplier": 0.4},
}


# ---------------------------------------------------------------------------
# Sidebar — Scenario Controls
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Environment Overrides")
        st.markdown(
            '<p style="color: #8892b0; font-size: 13px;">'
            "Adjust market conditions in real-time. The simulation picks up "
            "these values on every tick.</p>",
            unsafe_allow_html=True,
        )

        # Preset buttons
        st.markdown("#### 🎯 Quick Presets")
        for preset_name, preset_vals in SCENARIO_PRESETS.items():
            if st.button(preset_name, key=f"preset_{preset_name}", use_container_width=True):
                st.session_state["price_mult"] = preset_vals["price_multiplier"]
                st.session_state["demand_mult"] = preset_vals["demand_multiplier"]
                write_scenario_overrides(
                    preset_vals["price_multiplier"],
                    preset_vals["demand_multiplier"],
                )

        st.markdown("---")
        st.markdown("#### 🎚️ Fine-Tune Controls")

        # Initialize session state defaults
        if "price_mult" not in st.session_state:
            st.session_state["price_mult"] = 1.0
        if "demand_mult" not in st.session_state:
            st.session_state["demand_mult"] = 1.0

        price_mult = st.slider(
            "Price Multiplier",
            min_value=0.1,
            max_value=3.0,
            value=st.session_state["price_mult"],
            step=0.1,
            key="price_slider",
            help="Scales the base market price. >1.0 = expensive energy, <1.0 = cheap energy.",
        )
        demand_mult = st.slider(
            "Demand Multiplier",
            min_value=0.1,
            max_value=3.0,
            value=st.session_state["demand_mult"],
            step=0.1,
            key="demand_slider",
            help="Scales the base demand. >1.0 = high demand period, <1.0 = low demand.",
        )

        # Write overrides on every slider change
        write_scenario_overrides(price_mult, demand_mult)

        # Show current active scenario
        st.markdown("---")
        st.markdown("#### 📊 Active Scenario")

        price_color = "#ff1744" if price_mult > 1.5 else "#00c853" if price_mult < 0.7 else "#ffab00"
        demand_color = "#ff1744" if demand_mult > 1.5 else "#00c853" if demand_mult < 0.7 else "#ffab00"

        st.markdown(
            f'<div style="background: #1a1a2e; border-radius: 12px; padding: 16px; '
            f'border: 1px solid rgba(255,255,255,0.08);">'
            f'<p style="margin: 0 0 8px 0; color: #8892b0; font-size: 12px;">PRICE MULTIPLIER</p>'
            f'<p style="margin: 0 0 12px 0; color: {price_color}; font-size: 24px; font-weight: 700;">'
            f"{price_mult:.1f}x</p>"
            f'<p style="margin: 0 0 8px 0; color: #8892b0; font-size: 12px;">DEMAND MULTIPLIER</p>'
            f'<p style="margin: 0; color: {demand_color}; font-size: 24px; font-weight: 700;">'
            f"{demand_mult:.1f}x</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            '<p style="color: #8892b0; font-size: 11px;">'
            "💡 Try a <b>Price Spike</b> to see the agent switch to selling, "
            "or a <b>Demand Surge</b> to watch it stockpile energy.</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def render_header():
    st.markdown(
        """
        <style>
        .kpi-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px;
            padding: 22px 28px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        }
        .kpi-label {
            color: #8892b0;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 4px;
        }
        .kpi-value {
            font-size: 32px;
            font-weight: 700;
            color: #ccd6f6;
        }
        .kpi-value.positive { color: #00c853; }
        .kpi-value.negative { color: #ff1744; }
        .header-title {
            font-size: 28px;
            font-weight: 700;
            color: #ccd6f6;
            margin-bottom: 4px;
        }
        .header-sub {
            color: #8892b0;
            font-size: 14px;
        }
        .scenario-badge {
            display: inline-block;
            background: rgba(187,134,252,0.15);
            border: 1px solid #bb86fc;
            border-radius: 8px;
            padding: 4px 12px;
            color: #bb86fc;
            font-size: 12px;
            font-weight: 600;
            margin-left: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="header-title">⚡ Energy Trading Agent — Live Dashboard</p>'
        '<p class="header-sub">Reinforcement Learning PPO agent trading in a simulated energy market</p>',
        unsafe_allow_html=True,
    )


def render_kpis(df: pd.DataFrame):
    if df.empty:
        st.info("⏳ Waiting for simulation data…  Run `python scripts/run_simulation.py` to start.")
        return

    latest = df.iloc[-1]
    balance = latest["account_balance"]
    battery = latest["battery_level"]
    cum_profit = latest["cumulative_profit"]
    total_unmet = df["unmet_demand"].sum()
    num_buys = (df["action_name"] == "BUY").sum()
    num_sells = (df["action_name"] == "SELL").sum()

    profit_class = "positive" if cum_profit >= 0 else "negative"

    cols = st.columns(6)
    kpis = [
        ("Account Balance", f"${balance:,.2f}", ""),
        ("Battery Level", f"{battery:.1f} kWh", ""),
        ("Cumulative P&L", f"${cum_profit:,.2f}", profit_class),
        ("Buys", str(num_buys), ""),
        ("Sells", str(num_sells), ""),
        ("Unmet Demand", f"{total_unmet:.1f} kWh", "negative" if total_unmet > 0 else ""),
    ]
    for col, (label, value, cls) in zip(cols, kpis):
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value {cls}">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_price_demand_chart(df: pd.DataFrame):
    if df.empty:
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["sim_datetime"],
            y=df["price"],
            name="Price ($/kWh)",
            line=dict(color="#64ffda", width=2),
            fill="tozeroy",
            fillcolor="rgba(100,255,218,0.08)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["sim_datetime"],
            y=df["demand"],
            name="Demand (kWh)",
            line=dict(color="#ff6e40", width=2, dash="dot"),
        ),
        secondary_y=True,
    )

    for action_name, color in ACTION_COLORS.items():
        mask = df["action_name"] == action_name
        if mask.any():
            subset = df[mask]
            fig.add_trace(
                go.Scatter(
                    x=subset["sim_datetime"],
                    y=subset["price"],
                    mode="markers",
                    name=action_name,
                    marker=dict(
                        color=color,
                        size=10,
                        symbol=ACTION_SYMBOLS[action_name],
                        line=dict(width=1, color="white"),
                    ),
                ),
                secondary_y=False,
            )

    fig.update_layout(
        title="Market Price & Demand with Agent Actions",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=20, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text="Simulation Time", gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Price ($/kWh)", secondary_y=False, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Demand (kWh)", secondary_y=True, gridcolor="rgba(255,255,255,0.05)")

    st.plotly_chart(fig, use_container_width=True)


def render_battery_balance_chart(df: pd.DataFrame):
    if df.empty:
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["sim_datetime"],
            y=df["battery_level"],
            name="Battery (kWh)",
            line=dict(color="#bb86fc", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(187,134,252,0.10)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["sim_datetime"],
            y=df["account_balance"],
            name="Balance ($)",
            line=dict(color="#03dac6", width=2),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Battery Level & Account Balance Over Time",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=20, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text="Simulation Time", gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Battery (kWh)", secondary_y=False, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Balance ($)", secondary_y=True, gridcolor="rgba(255,255,255,0.05)")

    st.plotly_chart(fig, use_container_width=True)


def render_cumulative_profit_chart(df: pd.DataFrame):
    if df.empty:
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["sim_datetime"],
            y=df["cumulative_profit"],
            name="Cumulative P&L",
            line=dict(width=3),
            fill="tozeroy",
            fillcolor="rgba(0,200,83,0.12)",
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        title="Cumulative Profit & Loss",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=20, r=20, t=50, b=40),
        colorway=["#00c853"],
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Profit ($)", gridcolor="rgba(255,255,255,0.05)")

    st.plotly_chart(fig, use_container_width=True)


def render_action_log(df: pd.DataFrame):
    if df.empty:
        return
    st.subheader("📋 Recent Actions")
    display_cols = ["sim_datetime", "price", "demand", "action_name", "battery_level", "account_balance", "reward"]
    display = df[display_cols].tail(20).copy()
    display.columns = ["Time", "Price", "Demand", "Action", "Battery", "Balance", "Reward"]
    st.dataframe(display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    render_sidebar()
    render_header()

    df = load_log()

    render_kpis(df)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        render_price_demand_chart(df)
    with col2:
        render_battery_balance_chart(df)

    render_cumulative_profit_chart(df)

    render_action_log(df)

    # Auto-refresh
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()
