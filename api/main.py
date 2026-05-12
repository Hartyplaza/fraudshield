"""FraudShield FastAPI application."""

import json
import joblib
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.schemas import TransactionFeatures, FraudPredictionResponse
from src.preprocess import engineer_features

app = FastAPI(title="FraudShield AI", description="Real-time credit card fraud detection.", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model     = None
threshold = 0.5

@app.on_event("startup")
def load_ml_model():
    global model, threshold
    model_path     = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_pipeline.pkl')
    threshold_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'optimal_threshold.json')
    try:
        model = joblib.load(model_path)
        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                threshold = json.load(f).get('threshold', 0.5)
        logger.info(f"Model loaded. Threshold: {threshold:.3f}")
    except Exception as e:
        logger.warning(f"Model not found: {e}")

@app.get("/")
def root():
    return {"message": "FraudShield AI is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "threshold": threshold}

@app.post("/predict", response_model=FraudPredictionResponse)
def predict_fraud(transaction: TransactionFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        input_df = pd.DataFrame([transaction.model_dump()])
        input_df = engineer_features(input_df)
        prob     = model.predict_proba(input_df)[0][1]
        pred     = int(prob >= threshold)
        risk     = "High" if prob >= 0.7 else "Medium" if prob >= 0.3 else "Low"
        return FraudPredictionResponse(
            fraud_prediction=pred,
            fraud_probability=round(float(prob), 4),
            risk_level=risk,
            threshold_used=round(threshold, 4)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
