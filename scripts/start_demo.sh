#!/usr/bin/env bash
# ---------------------------------------------------------------
# start_demo.sh — Launch the full Energy Trading Agent demo
#
# This script starts three processes concurrently:
#   1. FastAPI Backend    (uvicorn)
#   2. Simulation Loop    (run_simulation.py)
#   3. Streamlit Dashboard
#
# Usage:
#   bash scripts/start_demo.sh            # default (1 sec = 1 min)
#   bash scripts/start_demo.sh --speed 60 # fast mode (1 sec = 1 hr)
#
# Press Ctrl-C to stop all processes.
# ---------------------------------------------------------------
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

SPEED="${1:-1}"  # default speed factor = 1 (1 sec = 1 min)

echo "============================================================"
echo "  ⚡  ENERGY TRADING AGENT — LIVE DEMO"
echo "============================================================"
echo ""

# Check for trained model
if [ ! -f "models/ppo_energy_agent.zip" ]; then
    echo "⚠️  No trained model found. Training agent first…"
    echo "   (This takes ~1-2 minutes)"
    echo ""
    uv run scripts/train_demo_agent.py --timesteps 200000
    echo ""
fi

# Clean previous log
rm -f data/demo_logs/simulation_log.csv

echo "🚀 Starting FastAPI backend on http://127.0.0.1:8000"
PYTHONPATH="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES= uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --log-level warning &
API_PID=$!
sleep 3  # give the server time to start

echo "📊 Starting simulation (speed=${SPEED}x) …"
PYTHONPATH="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES= uv run scripts/run_simulation.py --speed "$SPEED" --hours 168 &
SIM_PID=$!

echo "🖥️  Starting Streamlit dashboard …"
echo "   → Open http://localhost:8501 in your browser"
echo ""
PYTHONPATH="$PROJECT_ROOT" uv run streamlit run src/demo/dashboard.py --server.headless true --server.port 8501 &
DASH_PID=$!

# Trap Ctrl-C to kill all background processes
trap "echo ''; echo 'Shutting down…'; kill $API_PID $SIM_PID $DASH_PID 2>/dev/null; exit 0" INT TERM

echo ""
echo "============================================================"
echo "  All components running!"
echo "  Dashboard : http://localhost:8501"
echo "  API       : http://127.0.0.1:8000/health"
echo "  Press Ctrl-C to stop everything."
echo "============================================================"
echo ""

# Wait for simulation to finish (or Ctrl-C)
wait $SIM_PID 2>/dev/null

echo ""
echo "Simulation finished. Dashboard is still running."
echo "Press Ctrl-C to stop the dashboard and API."
wait
