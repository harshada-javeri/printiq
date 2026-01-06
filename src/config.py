"""
Configuration for PrintIQ ML models.

Centralized settings for reproducibility and easy tuning.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Data
RAW_DATA_FILE = RAW_DATA_DIR / "print_jobs.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

# Model artifacts
FAILURE_MODEL_FILE = MODELS_DIR / "failure_model.pkl"
QUALITY_MODEL_FILE = MODELS_DIR / "quality_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
FAILURE_EXPLAINER_FILE = MODELS_DIR / "failure_explainer.pkl"
QUALITY_EXPLAINER_FILE = MODELS_DIR / "quality_explainer.pkl"

# Features
FEATURE_SET = [
    "printer_age",
    "head_type",
    "ink_viscosity",
    "paper_gsm",
    "humidity",
    "temperature",
    "coverage_pct",
    "nozzles_clean",
]

CATEGORICAL_FEATURES = ["head_type"]
NUMERIC_FEATURES = [f for f in FEATURE_SET if f not in CATEGORICAL_FEATURES]

# Targets
FAILURE_TARGET = "failed"
QUALITY_TARGET = "quality_score"

# Model hyperparameters
MODEL_CONFIG = {
    "failure_model": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "quality_model": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
}

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
