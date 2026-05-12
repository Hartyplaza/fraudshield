# FraudShield AI — Credit Card Fraud Detection

An end-to-end machine learning system for real-time credit card fraud detection. Built with multiple detection strategies, cost-sensitive threshold optimisation, SHAP explainability, FastAPI, and an enterprise-grade Streamlit dashboard.

**Live Demo:** https://fraudshield-9tyavaubm563jwkabq8gur.streamlit.app/

---

## Problem Statement

Credit card fraud costs the global economy over $30 billion annually. The core challenge is not building a model — it is building one that works under extreme conditions:

- Only **0.17% of transactions are fraud** — a 577:1 class imbalance
- Standard accuracy metrics are useless — predicting everything as legitimate gives 99.83% accuracy while catching zero fraud
- **Missing a fraud costs 100x more** than a false alarm
- Models must explain why a transaction was flagged for investigators

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 |
| Fraud cases | 492 (0.17%) |
| Imbalance ratio | 577:1 |
| Features | V1–V28 (PCA-anonymized), Time, Amount |
| Time span | 48 hours of European cardholder transactions |

---

## Results

### Model Comparison (sorted by PR-AUC)

| Model | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| **Random Forest** | **0.8303** | **0.9669** | **0.7895** | **0.9375** | **0.8571** |
| XGBoost + SMOTETomek | 0.7714 | 0.9640 | 0.8632 | 0.0756 | 0.1390 |
| XGBoost | 0.7340 | 0.9565 | 0.8842 | 0.0626 | 0.1169 |
| Logistic Regression | 0.6911 | 0.9655 | 0.8737 | 0.0550 | 0.1035 |
| Isolation Forest | 0.1464 | 0.9371 | 0.2632 | 0.2294 | 0.2451 |

> PR-AUC is the primary metric for fraud detection. With extreme class imbalance, ROC-AUC is misleading — a random classifier scores 0.5 on ROC-AUC but only 0.0017 on PR-AUC.

### Cost Analysis (FN = €100, FP = €1)

| Model | Threshold | TP | FN | FP | Total Cost |
|---|---|---|---|---|---|
| Random Forest | Optimised | 83 | 12 | 172 | **€1,372** |
| Random Forest | Default (0.5) | 75 | 20 | 5 | €2,005 |
| XGBoost + SMOTETomek | Default (0.5) | 82 | 13 | 1,003 | €2,303 |
| XGBoost | Default (0.5) | 84 | 11 | 1,258 | €2,358 |
| Logistic Regression | Default (0.5) | 83 | 12 | 1,426 | €2,626 |

> Threshold optimisation reduced total business cost by **31%** for Random Forest (€2,005 → €1,372).

---

## Project Structure

```
fraudshield/
├── .streamlit/
│   └── config.toml              # Dark theme configuration
├── data/
│   └── raw/                     # creditcard.csv (not tracked — too large)
├── notebooks/
│   ├── 01_eda.ipynb             # Class imbalance, feature distributions, t-SNE
│   ├── 02_feature_engineering.ipynb  # Scaling, resampling strategy comparison
│   ├── 03_modeling.ipynb        # 4 models, threshold optimisation, cost analysis
│   └── 04_explainability.ipynb  # SHAP values, waterfall plots, dependence plots
├── src/
│   ├── config.py                # Paths, cost parameters, MLflow config
│   ├── preprocess.py            # RobustScaler, feature engineering pipeline
│   ├── train.py                 # Training loop with MLflow tracking
│   └── predict.py               # Inference with optimal threshold
├── api/
│   ├── main.py                  # FastAPI application
│   └── schemas.py               # Pydantic request/response models
├── app/
│   └── streamlit_app.py         # 6-page Streamlit dashboard
├── models/
│   ├── best_model_pipeline.pkl  # Trained XGBoost pipeline
│   ├── optimal_threshold.json   # Optimised threshold (0.072)
│   └── feature_names.json       # Feature metadata
├── runtime.txt                  # Python 3.11 for Streamlit Cloud
└── requirements.txt
```

---

## Notebooks

### 01 — EDA
- Class imbalance analysis (577:1 ratio, log-scale visualisation)
- V feature distributions comparing fraud vs legitimate
- Time and Amount analysis — fraud peaks at low-traffic hours
- Correlation analysis and PCA scatter plots
- t-SNE visualisation showing distinct fraud clusters

### 02 — Feature Engineering
- 5 new features from Time and Amount: `hour`, `day`, `log_amount`, `amount_bin`, `high_amount_night`
- RobustScaler vs StandardScaler comparison — RobustScaler handles fraud outliers better
- Three resampling strategies compared on a 10% sample: SMOTE, ADASYN, SMOTETomek

### 03 — Modeling
- Logistic Regression with `class_weight='balanced'`
- Random Forest with SMOTE resampling
- XGBoost with ADASYN and `scale_pos_weight`
- Isolation Forest — unsupervised anomaly detection (no labels used)
- Threshold optimisation minimising `FN × 100 + FP × 1`
- Full cost-sensitive business impact analysis

### 04 — Explainability
- SHAP TreeExplainer on XGBoost
- Global feature importance (mean |SHAP|)
- Waterfall plots for highest-confidence fraud and most clearly legitimate transactions
- Fraud vs Legitimate average SHAP comparison
- Feature dependence plots for top 4 features

---

## Engineered Features

| Feature | Formula | Importance |
|---|---|---|
| `hour` | `(Time // 3600) % 24` | Medium |
| `day` | `(Time // 86400).astype(int)` | Low |
| `log_amount` | `np.log1p(Amount)` | High |
| `amount_bin` | `pd.cut(Amount, bins=[0,10,50,200,1000,inf])` | Medium |
| `high_amount_night` | `(Amount > 200) & (hour < 6)` | High |

---

## Dashboard Pages

| Page | Description |
|---|---|
| Overview | Project summary, pipeline steps, problem statement, tech stack |
| Analytics | Class distribution, fraud by hour, SHAP importance, amount stats |
| Transaction Analysis | Real-time fraud scoring form for all 28 V features + Time + Amount |
| Live Feed | Simulated batch transaction monitoring with refresh and risk breakdown |
| Features | Engineered feature explanations with formulas and impact charts |
| Model Performance | Radar chart, metrics comparison, leaderboard, confusion matrix |

---

## API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Model status and threshold |
| POST | `/predict` | Score a single transaction |

Example request:

```python
import requests

transaction = {
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Time": 50000.0,
    "Amount": 149.62
}
response = requests.post("http://localhost:8000/predict", json=transaction)
print(response.json())
```

---

## Quickstart

```bash
git clone https://github.com/Hartyplaza/fraudshield.git
cd fraudshield
pip install -r requirements.txt

# Download dataset from Kaggle and place in data/raw/creditcard.csv
# Then run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, NumPy |
| ML | scikit-learn, XGBoost |
| Imbalance | imbalanced-learn (SMOTE, ADASYN, SMOTETomek) |
| Explainability | SHAP |
| Tracking | MLflow |
| API | FastAPI, uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Cloud |

---

## Author

**Ofigwe Hart** — Data Scientist / ML Engineer

- LinkedIn: [linkedin.com/in/hart-ofigwe](https://www.linkedin.com/in/hart-ofigwe)
- GitHub: [github.com/Hartyplaza](https://github.com/Hartyplaza)
- Live Demo: [fraudshield.streamlit.app](https://fraudshield-9tyavaubm563jwkabq8gur.streamlit.app/)
