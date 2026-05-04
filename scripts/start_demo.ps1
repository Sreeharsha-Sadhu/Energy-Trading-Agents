$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$env:PYTHONPATH = $ProjectRoot
$env:PYTHONIOENCODING = "utf-8"
$Speed = if ($args.Length -gt 0) { $args[0] } else { 1 }

Write-Host "============================================================"
Write-Host "  ⚡  ENERGY TRADING AGENT — LIVE DEMO"
Write-Host "============================================================"
Write-Host ""

if (!(Test-Path "models\ppo_energy_agent.zip")) {
    Write-Host "⚠️  No trained model found. Training agent first…"
    Write-Host '   (This takes ~1-2 minutes)'
    Write-Host ""
    uv run scripts/train_demo_agent.py --timesteps 200000
    Write-Host ""
}

Remove-Item -Path "data\demo_logs\simulation_log.csv" -ErrorAction SilentlyContinue

Write-Host "🚀 Starting FastAPI backend on http://127.0.0.1:8000"
$ApiProcess = Start-Process -NoNewWindow -PassThru -FilePath "uv" -ArgumentList "run uvicorn src.main:app --host 127.0.0.1 --port 8000 --log-level warning"
Start-Sleep -Seconds 3

Write-Host "📊 Starting simulation (speed=${Speed}x) …"
$SimProcess = Start-Process -NoNewWindow -PassThru -FilePath "uv" -ArgumentList "run scripts/run_simulation.py --speed $Speed --hours 8760"

Write-Host "🖥️  Starting Streamlit dashboard …"
Write-Host "   → Open http://localhost:8501 in your browser"
Write-Host ""
$DashProcess = Start-Process -NoNewWindow -PassThru -FilePath "uv" -ArgumentList "run streamlit run src/demo/dashboard.py --server.headless true --server.port 8501"

Write-Host "============================================================"
Write-Host "  All components running in background!"
Write-Host "  Dashboard : http://localhost:8501"
Write-Host "  API       : http://127.0.0.1:8000/health"
Write-Host "  To stop them, run: Stop-Process -Id $($ApiProcess.Id), $($SimProcess.Id), $($DashProcess.Id)"
Write-Host "============================================================"
