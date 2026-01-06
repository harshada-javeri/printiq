"""
Model training pipeline for failure and quality predictions.

Usage:
    python src/train.py
"""

import logging
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

from .config import (
    RAW_DATA_FILE,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    FAILURE_MODEL_FILE,
    QUALITY_MODEL_FILE,
    PREPROCESSOR_FILE,
    FEATURE_SET,
    FAILURE_TARGET,
    QUALITY_TARGET,
    MODEL_CONFIG,
    TEST_SIZE,
    RANDOM_STATE,
)
from .features import FeaturePreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_split_data(random_state: int = RANDOM_STATE) -> tuple:
    """
    Load raw data and create train/test split.
    
    Args:
        random_state: Seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train_failure, y_test_failure, 
                  y_train_quality, y_test_quality, df_train, df_test)
    """
    logger.info(f"Loading data from {RAW_DATA_FILE}")
    
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found: {RAW_DATA_FILE}\n"
            "Run: python data/generate_data.py"
        )
    
    df = pd.read_csv(RAW_DATA_FILE)
    logger.info(f"Loaded {len(df):,} samples with {len(df.columns)} columns")
    
    # Split features and targets
    X = df[FEATURE_SET].copy()
    y_failure = df[FAILURE_TARGET].copy()
    y_quality = df[QUALITY_TARGET].copy()
    
    # Train/test split
    X_train, X_test, y_failure_train, y_failure_test, y_quality_train, y_quality_test = train_test_split(
        X, y_failure, y_quality,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y_failure  # Stratify by failure class
    )
    
    logger.info(f"Train set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")
    logger.info(f"Failure rate: {y_failure_train.mean()*100:.1f}%")
    
    # Save splits
    PROCESSED_DATA_DIR = TRAIN_DATA_FILE.parent
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    df_train = X_train.copy()
    df_train[FAILURE_TARGET] = y_failure_train
    df_train[QUALITY_TARGET] = y_quality_train
    df_train.to_csv(TRAIN_DATA_FILE, index=False)
    
    df_test = X_test.copy()
    df_test[FAILURE_TARGET] = y_failure_test
    df_test[QUALITY_TARGET] = y_quality_test
    df_test.to_csv(TEST_DATA_FILE, index=False)
    
    logger.info(f"Saved training data to {TRAIN_DATA_FILE}")
    logger.info(f"Saved test data to {TEST_DATA_FILE}")
    
    return X_train, X_test, y_failure_train, y_failure_test, y_quality_train, y_quality_test


def train_failure_model(X_train, y_train) -> RandomForestClassifier:
    """
    Train failure prediction model (binary classification).
    
    Args:
        X_train: Training features (preprocessed)
        y_train: Training failure labels
        
    Returns:
        Trained RandomForestClassifier
    """
    logger.info("Training failure prediction model...")
    
    model = RandomForestClassifier(**MODEL_CONFIG["failure_model"])
    model.fit(X_train, y_train)
    
    # Report feature importance
    importances = model.feature_importances_
    for feat, imp in zip(FEATURE_SET, importances):
        logger.info(f"  {feat}: {imp:.4f}")
    
    return model


def train_quality_model(X_train, y_train) -> RandomForestRegressor:
    """
    Train quality prediction model (regression).
    
    Args:
        X_train: Training features (preprocessed)
        y_train: Training quality scores
        
    Returns:
        Trained RandomForestRegressor
    """
    logger.info("Training quality prediction model...")
    
    model = RandomForestRegressor(**MODEL_CONFIG["quality_model"])
    model.fit(X_train, y_train)
    
    # Report feature importance
    importances = model.feature_importances_
    for feat, imp in zip(FEATURE_SET, importances):
        logger.info(f"  {feat}: {imp:.4f}")
    
    return model


def save_artifacts(preprocessor, failure_model, quality_model):
    """Save trained models and preprocessor to disk."""
    MODELS_DIR = FAILURE_MODEL_FILE.parent
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    joblib.dump(failure_model, FAILURE_MODEL_FILE)
    joblib.dump(quality_model, QUALITY_MODEL_FILE)
    
    logger.info(f"Saved preprocessor to {PREPROCESSOR_FILE}")
    logger.info(f"Saved failure model to {FAILURE_MODEL_FILE}")
    logger.info(f"Saved quality model to {QUALITY_MODEL_FILE}")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("PrintIQ Model Training Pipeline")
    logger.info("=" * 60)
    
    # Load and prepare data
    X_train, X_test, y_failure_train, y_failure_test, y_quality_train, y_quality_test = (
        load_and_split_data()
    )
    
    # Preprocess features
    logger.info("Fitting feature preprocessor...")
    preprocessor = FeaturePreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train models
    failure_model = train_failure_model(X_train_processed, y_failure_train)
    quality_model = train_quality_model(X_train_processed, y_quality_train)
    
    # Evaluate on test set (quick check)
    failure_acc = failure_model.score(X_test_processed, y_failure_test)
    quality_r2 = quality_model.score(X_test_processed, y_quality_test)
    
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Failure Model Accuracy: {failure_acc:.4f}")
    logger.info(f"  Quality Model R² Score: {quality_r2:.4f}")
    
    # Save artifacts
    save_artifacts(preprocessor, failure_model, quality_model)
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
