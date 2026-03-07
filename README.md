# **Energy Load Forecasting with RL Agent Integration**

A complete end-to-end machine learning system for forecasting energy load across residential and industrial consumer segments.
This repository includes:

* A **synthetic energy load generator** tailored for reinforcement learning environments
* A full **preprocessing, feature engineering, and segmentation pipeline**
* **Model training**, **cross-validated hyperparameter tuning**, and **comprehensive evaluation**
* **Model comparison, inference, automated CV plots, and unified experiment reports**
* Cleanly refactored and modular Python code extracted from the original notebooks

---

## 📘 **About the Notebooks**

This project was initially developed through a sequence of analytical Jupyter notebooks.
They contain:

* Exploratory analysis
* Visual insights
* Intermediate outputs
* Tuning experiments
* Interpretations

To make the project production-ready, **all executable logic has been extracted into the `src/` Python modules**, while the notebooks remain **unchanged** for transparency and learning value.

**The notebooks serve as documentation, not as the production codebase.**
All refactored `.py` modules were generated directly from notebook logic, cleaned, modularized, and optimized for reproducibility.

---

## 📁 **Project Structure**

```
.
├── data/
│   ├── processed/                 # Fully processed dataset
│   ├── segments/                  # Six modeling segments
│   ├── demo_logs/                 # Live demo simulation logs
│   └── san_diego_energy_load_data.csv
│
├── notebooks/                     # Original notebooks (unchanged)
│   ├── pre-processing.ipynb
│   ├── EDA.ipynb
│   ├── modelTuning.ipynb
│   └── modelComparison.ipynb
│
├── src/
│   ├── forecaster/                # Synthetic generator & load models
│   ├── demo/                      # Streamlit dashboard & data provider
│   ├── agent/                     # PPO model training & inference
│   ├── envs/                      # Gymnasium trading environment
│   ├── api/                       # FastAPI routes & schemas
│   └── main.py                    # FastAPI application entrypoint
│
├── scripts/                       # Automation scripts
│   ├── start_demo.sh              # One-command demo launcher
│   ├── train_demo_agent.py        # Train PPO agent
│   ├── run_simulation.py          # Market simulation loop
│   ├── generate_dataset.py
│   ├── preprocess_data.py
│   ├── train_all_models.py
│   ├── tune_all_models.py
│   ├── generate_cv_plots.py
│   └── generate_experiment_report.py
│
├── models/                        # Saved model artifacts
├── plots/                         # Auto-generated CV visualizations
├── reports/                       # Unified experiment report(s)
├── Docs/                          # Documentation (incl. DEMO_GUIDE.md)
├── pyproject.toml
└── README.md
```

---

## 🔧 **Key Features**

### **1. Synthetic Data Generator**

Generates multi-year hourly data reflecting:

* Industry and residential consumption patterns
* Solar generation effects
* Seasonal and daily cycles
* Weather simulations
* Meter counts and load growth
* Loss adjustments
* Features seamless integration points for RL Agents for demand response optimization

---

### **2. Automated Preprocessing Pipeline**

Includes:

* Timestamp creation, sorting, and cleaning
* Prioritizing “Final” submissions over “Initial”
* Synthetic weather data (temperature & irradiance)
* Time features (hour, weekday, month, quarter, etc.)
* Lag + rolling window features
* Segmentation into six modeling groups

Outputs:

```
data/processed/processed_full.csv
data/segments/{Segment}.csv
```

---

### **3. Modeling Pipeline**

Supports:

* **LightGBM**
* **XGBoost**
* **TimeSeriesSplit cross-validation**
* **Optuna hyperparameter tuning (with CV)**
* **Full training automation**

Champion models are automatically saved to:

```
models/{Segment}_{ModelType}_best.joblib
```

---

## 📉 **Cross-Validation & Diagnostics**

This project implements **TimeSeriesSplit cross-validation** for both baseline training and Optuna tuning.

### Includes auto-generated plots:

1. **Fold-wise CV performance curves**
2. **Model comparison bar charts (LightGBM vs XGBoost)**
3. **Optuna trial history**
4. **Parameter importance via Optuna**

Run:

```
uv run scripts/generate_cv_plots.py
```

Plots saved to:

```
plots/cv_plots/
```

These diagnostics reveal:

* Temporal drift
* Model stability
* Hyperparameter sensitivity
* Overfitting / underfitting

---

## 📊 **Unified Experiment Report**

A single consolidated report with:

* Segment-wise summaries
* CV metrics
* Best model selection
* Embedded CV plots
* Linked artifacts

Generate it with:

```
uv run scripts/generate_experiment_report.py
```

Output:

```
reports/experiment_report.md
```

---

## ⚡ **Live Demo — RL Trading Agent**

A real-time Streamlit dashboard that visualises a PPO agent buying, selling, and holding energy in a simulated market.

### Quick start

```bash
bash scripts/start_demo.sh
```

This launches the FastAPI backend, simulation loop, and Streamlit dashboard. Open **http://localhost:8501** to watch the agent trade.

For full setup instructions, architecture details, configuration, and troubleshooting see
**[Docs/DEMO_GUIDE.md](Docs/DEMO_GUIDE.md)**.

---

## 🚀 **Quick Start**

### **1. Install dependencies**

```
uv sync
```

---

### **2. Generate synthetic dataset**

```
uv run scripts/generate_dataset.py --full
```

---

### **3. Preprocess & segment**

```
uv run scripts/preprocess_data.py --force
```

---

### **4. Train models**

```
uv run scripts/train_all_models.py
```

---

### **5. Hyperparameter tuning**

```
uv run scripts/tune_all_models.py --n_trials 40
```

---

### **6. Run inference**

```
uv run scripts/run_inference.py \
  --model models/Residential_Solar_LightGBM_best.joblib \
  --input data/segments/Residential_Solar.csv \
  --out predictions.csv
```

---

## 🧪 **Notebooks (Unmodified)**

The following notebooks remain unaltered and serve as the documented research and development process:

* `pre-processing.ipynb`
* `EDA.ipynb`
* `modelTuning.ipynb`
* `modelComparison.ipynb`

They are preserved with outputs for transparency and education.

---

## 📂 **Modeling Segments**

Six independent segments:

1. Residential_Non-Solar
2. Residential_Solar
3. Small_Scale_Industries_Non-Solar
4. Small_Scale_Industries_Solar
5. Medium_Scale_Industries_Non-Solar
6. Medium_Scale_Industries_Solar

Each has its own model, tuning history, and evaluation.

---

## 📝 **Requirements**

* Python 3.12+
* Pandas, NumPy
* LightGBM, XGBoost
* Optuna, Joblib, Matplotlib
* Streamlit, Plotly
* Stable-Baselines3, Gymnasium
* FastAPI, Uvicorn
