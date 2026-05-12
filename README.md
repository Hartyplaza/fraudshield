# FraudShield AI — Credit Card Fraud Detection

An end-to-end machine learning system for detecting credit card fraud in real-time. Built with multiple fraud detection strategies, threshold optimization, and cost-sensitive evaluation.

---

## Project Overview

Credit card fraud costs the global economy over $30 billion annually. This project builds a production-ready fraud detection system that:

- Handles **extreme class imbalance** (0.17% fraud rate)
- Compares **5 detection strategies** including anomaly detection
- Optimizes decision **threshold for business cost**
- Explains predictions with **SHAP values**
- Serves real-time predictions via **FastAPI**
- Provides an interactive **Streamlit dashboard**

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 492 fraud (0.17%)
- **Features:** V1–V28 (PCA-anonymized), Time, Amount, Class

---

## Results

| Model | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| XGBoost (Tuned) | TBD | TBD | TBD | TBD |
| Isolation Forest | TBD | TBD | TBD | TBD |

---

## Project Structure

```
fraudshield/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/                   # creditcard.csv
│   └── processed/             # train/test splits
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── api/
│   ├── main.py
│   └── schemas.py
├── app/
│   └── streamlit_app.py
├── models/
├── mlflow/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/Hartyplaza/fraudshield.git
cd fraudshield
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

---

## Author

**Ofigwe Hart** — Data Scientist / ML Engineer
- LinkedIn: [linkedin.com/in/hart-ofigwe](https://www.linkedin.com/in/hart-ofigwe)
- GitHub: [github.com/Hartyplaza](https://github.com/Hartyplaza)
