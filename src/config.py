"""
Project-wide configuration and paths.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent.parent
DATA_DIR           = BASE_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR         = BASE_DIR / "models"
MLFLOW_DIR         = BASE_DIR / "mlflow"

# ── Data ───────────────────────────────────────────────────────────────────────
RAW_DATA_FILE       = RAW_DATA_DIR / "creditcard.csv"
PROCESSED_TRAIN     = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST      = PROCESSED_DATA_DIR / "test.csv"

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_FILE          = MODELS_DIR / "best_model_pipeline.pkl"
THRESHOLD_FILE      = MODELS_DIR / "optimal_threshold.json"

TARGET_COLUMN       = "Class"
TEST_SIZE           = 0.2
RANDOM_STATE        = 42

# ── Fraud Cost Parameters (for threshold optimization) ─────────────────────────
# Cost of missing a fraud (false negative) vs cost of false alarm (false positive)
COST_FN             = 100   # Cost of missing a fraud (€100 average loss)
COST_FP             = 1     # Cost of false alarm (€1 investigation cost)

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = str(MLFLOW_DIR)
EXPERIMENT_NAME     = "fraudshield"

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST            = os.getenv("API_HOST", "0.0.0.0")
API_PORT            = int(os.getenv("API_PORT", 8000))
