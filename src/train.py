"""
Model training with multiple fraud detection strategies.
Includes threshold optimization for business cost minimization.
"""

import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    f1_score, recall_score, precision_score, roc_auc_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from loguru import logger

from src.config import (
    RAW_DATA_FILE, MODEL_FILE, THRESHOLD_FILE,
    MLFLOW_TRACKING_URI, EXPERIMENT_NAME,
    TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
    COST_FN, COST_FP
)
from src.preprocess import load_raw_data, engineer_features, get_feature_groups, build_preprocessor


def optimize_threshold(y_true, y_prob, cost_fn=COST_FN, cost_fp=COST_FP):
    """
    Find optimal classification threshold that minimizes business cost.
    cost_fn = cost of missing a fraud (false negative)
    cost_fp = cost of false alarm (false positive)
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_threshold = 0.5
    best_cost = float('inf')

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fn = ((y_true == 1) & (y_pred == 0)).sum()  # missed frauds
        fp = ((y_true == 0) & (y_pred == 1)).sum()  # false alarms
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    return best_threshold, best_cost


def evaluate_model(pipeline, X_test, y_test, threshold=0.5):
    """Evaluate model with fraud-appropriate metrics."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'PR-AUC':    round(average_precision_score(y_test, y_prob), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, y_prob), 4),
        'Recall':    round(recall_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'F1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
    }


def train():
    # ── Load & prep ────────────────────────────────────────────────────────────
    df = load_raw_data()
    df = engineer_features(df)
    X  = df.drop(columns=[TARGET_COLUMN])
    y  = df[TARGET_COLUMN]

    numeric_cols, categorical_cols = get_feature_groups(df.drop(columns=[TARGET_COLUMN]))
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Fraud rate: {y_train.mean():.4%}")

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    models = {
        'logistic_regression': (
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
            SMOTE(random_state=RANDOM_STATE)
        ),
        'random_forest': (
            RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=-1),
            SMOTE(random_state=RANDOM_STATE)
        ),
        'xgboost': (
            XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                          scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                          eval_metric='aucpr', random_state=RANDOM_STATE, verbosity=0),
            ADASYN(random_state=RANDOM_STATE)
        ),
        'xgboost_smotetomek': (
            XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                          scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                          eval_metric='aucpr', random_state=RANDOM_STATE, verbosity=0),
            SMOTETomek(random_state=RANDOM_STATE)
        ),
    }

    best_pipeline  = None
    best_pr_auc    = 0.0
    best_name      = ""
    best_threshold = 0.5

    for name, (clf, sampler) in models.items():
        with mlflow.start_run(run_name=name):
            logger.info(f"Training: {name}")

            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('sampler',      sampler),
                ('classifier',   clf),
            ])
            pipeline.fit(X_train, y_train)

            # Optimize threshold for business cost
            y_prob     = pipeline.predict_proba(X_test)[:, 1]
            threshold, cost = optimize_threshold(y_test, y_prob)
            metrics    = evaluate_model(pipeline, X_test, y_test, threshold)

            mlflow.log_params({'model': name, 'threshold': threshold})
            mlflow.log_metrics({**metrics, 'business_cost': cost})
            mlflow.sklearn.log_model(pipeline, artifact_path='model')

            logger.info(f"{name} → PR-AUC: {metrics['PR-AUC']} | Recall: {metrics['Recall']} | Threshold: {threshold:.3f}")

            if metrics['PR-AUC'] > best_pr_auc:
                best_pr_auc    = metrics['PR-AUC']
                best_pipeline  = pipeline
                best_name      = name
                best_threshold = threshold

    # ── Save ───────────────────────────────────────────────────────────────────
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_FILE)

    with open(THRESHOLD_FILE, 'w') as f:
        json.dump({'threshold': best_threshold, 'model': best_name}, f)

    logger.info(f"Best model: {best_name} (PR-AUC={best_pr_auc:.4f}, threshold={best_threshold:.3f})")
    return best_pipeline, best_threshold


if __name__ == "__main__":
    train()
