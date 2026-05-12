"""
Inference logic for fraud detection.
"""

import json
import joblib
import pandas as pd
import numpy as np
from loguru import logger

from src.config import MODEL_FILE, THRESHOLD_FILE
from src.preprocess import engineer_features


def load_model():
    model     = joblib.load(MODEL_FILE)
    threshold = 0.5
    if THRESHOLD_FILE.exists():
        with open(THRESHOLD_FILE) as f:
            threshold = json.load(f).get('threshold', 0.5)
    logger.info(f"Model loaded. Threshold: {threshold:.3f}")
    return model, threshold


def predict(data: dict | pd.DataFrame, model=None, threshold=0.5):
    """Predict fraud for a single transaction or batch."""
    if model is None:
        model, threshold = load_model()

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    df = engineer_features(df)
    probs      = model.predict_proba(df)[:, 1]
    predictions = (probs >= threshold).astype(int)

    return {
        'fraud_prediction': predictions.tolist(),
        'fraud_probability': probs.tolist(),
        'risk_level': ['High' if p >= 0.7 else 'Medium' if p >= 0.3 else 'Low'
                       for p in probs],
    }
