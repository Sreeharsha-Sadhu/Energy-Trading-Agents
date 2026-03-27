import joblib
import pandas as pd
from src.forecaster.modeling.inference import get_segment_model_path, predict
from src.demo.data_provider import generate_hourly_timeline

segment = "Residential_Solar"
model_path = get_segment_model_path(segment)
print("model_path:", model_path)
model = joblib.load(str(model_path))
print("model loaded ok")

df = generate_hourly_timeline(pd.Timestamp.now(), num_hours=48)
print("timeline shape:", df.shape)

preds = predict(model, df)
print("preds type:", type(preds))
try:
    print("preds len:", len(preds))
    print("sample preds:", list(preds[:5]))
except Exception as e:
    print("preds repr:", repr(preds))
print("done")
