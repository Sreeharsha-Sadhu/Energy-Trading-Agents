import pandas as pd
import numpy as np

df = pd.read_csv("data/demo_logs/simulation_log.csv", parse_dates=["sim_datetime"])
# coerce numeric
for c in ["actual_demand", "predicted_demand", "forecast_error"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

n = len(df)
clip_val = float(0.1)
percent_clipped = (df["predicted_demand"] == clip_val).sum() / n * 100
mae = df["forecast_error"].abs().mean()
mape = (
    df["forecast_error"].abs() / df["actual_demand"].replace(0, np.nan)
).dropna().mean() * 100

top10 = (
    df.assign(abs_err=df["forecast_error"].abs())
    .sort_values("abs_err", ascending=False)
    .head(10)
)

print(f"rows={n}")
print(f"percent_clipped={percent_clipped:.2f}%")
print(f"MAE={mae:.4f}")
print(f"MAPE={mape:.2f}%")
print("\nTop-10 errors:")
print(
    top10[
        ["sim_datetime", "actual_demand", "predicted_demand", "forecast_error"]
    ].to_string(index=False)
)
