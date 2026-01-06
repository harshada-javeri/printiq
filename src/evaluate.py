"""
Model evaluation and performance metrics.

Usage:
    python src/evaluate.py
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
import joblib

from .config import (
    TEST_DATA_FILE,
    PREPROCESSOR_FILE,
    FAILURE_MODEL_FILE,
    QUALITY_MODEL_FILE,
    FEATURE_SET,
    FAILURE_TARGET,
    QUALITY_TARGET,
)
from .features import FeaturePreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_artifacts():
    """Load trained models and preprocessor."""
    if not all([
        PREPROCESSOR_FILE.exists(),
        FAILURE_MODEL_FILE.exists(),
        QUALITY_MODEL_FILE.exists(),
    ]):
        raise FileNotFoundError(
            "Model artifacts not found. Run: python src/train.py"
        )
    
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    failure_model = joblib.load(FAILURE_MODEL_FILE)
    quality_model = joblib.load(QUALITY_MODEL_FILE)
    
    return preprocessor, failure_model, quality_model


def evaluate_failure_model(y_true, y_pred, y_pred_proba):
    """
    Evaluate binary classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted classes
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dict of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": confusion_matrix(y_true, y_pred)[0, 0],
        "fp": confusion_matrix(y_true, y_pred)[0, 1],
        "fn": confusion_matrix(y_true, y_pred)[1, 0],
        "tp": confusion_matrix(y_true, y_pred)[1, 1],
    }
    
    # Add specificity
    metrics["specificity"] = metrics["tn"] / (metrics["tn"] + metrics["fp"] + 1e-8)
    
    return metrics


def evaluate_quality_model(y_true, y_pred):
    """
    Evaluate regression model.
    
    Args:
        y_true: True quality scores
        y_pred: Predicted quality scores
        
    Returns:
        Dict of metrics
    """
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
    }
    
    return metrics


def main():
    """Main evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("PrintIQ Model Evaluation")
    logger.info("=" * 60)
    
    # Load test data
    if not TEST_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Test data not found: {TEST_DATA_FILE}\n"
            "Run: python src/train.py"
        )
    
    df_test = pd.read_csv(TEST_DATA_FILE)
    X_test = df_test[FEATURE_SET]
    y_failure_test = df_test[FAILURE_TARGET]
    y_quality_test = df_test[QUALITY_TARGET]
    
    logger.info(f"Test set: {len(df_test):,} samples")
    
    # Load models
    preprocessor, failure_model, quality_model = load_artifacts()
    
    # Preprocess test features
    X_test_processed = preprocessor.transform(X_test)
    
    # === Failure Model Evaluation ===
    logger.info("\n" + "=" * 60)
    logger.info("FAILURE PREDICTION MODEL")
    logger.info("=" * 60)
    
    y_failure_pred = failure_model.predict(X_test_processed)
    y_failure_proba = failure_model.predict_proba(X_test_processed)[:, 1]
    
    failure_metrics = evaluate_failure_model(y_failure_test, y_failure_pred, y_failure_proba)
    
    logger.info(f"Accuracy:  {failure_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {failure_metrics['precision']:.4f}")
    logger.info(f"Recall:    {failure_metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {failure_metrics['f1']:.4f}")
    logger.info(f"Specificity: {failure_metrics['specificity']:.4f}")
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Negatives:  {int(failure_metrics['tn'])}")
    logger.info(f"  False Positives: {int(failure_metrics['fp'])}")
    logger.info(f"  False Negatives: {int(failure_metrics['fn'])}")
    logger.info(f"  True Positives:  {int(failure_metrics['tp'])}")
    
    # === Quality Model Evaluation ===
    logger.info("\n" + "=" * 60)
    logger.info("QUALITY PREDICTION MODEL")
    logger.info("=" * 60)
    
    y_quality_pred = quality_model.predict(X_test_processed)
    quality_metrics = evaluate_quality_model(y_quality_test, y_quality_pred)
    
    logger.info(f"MAE (Mean Absolute Error):  {quality_metrics['mae']:.4f}")
    logger.info(f"RMSE (Root Mean Squared Error): {quality_metrics['rmse']:.4f}")
    logger.info(f"R² Score: {quality_metrics['r2']:.4f}")
    logger.info(f"MAPE: {quality_metrics['mape']:.2f}%")
    
    # === Feature Importance ===
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 60)
    
    logger.info("\nFailure Model:")
    failure_importances = sorted(
        zip(FEATURE_SET, failure_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for feat, imp in failure_importances:
        logger.info(f"  {feat:20s}: {imp:.4f}")
    
    logger.info("\nQuality Model:")
    quality_importances = sorted(
        zip(FEATURE_SET, quality_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for feat, imp in quality_importances:
        logger.info(f"  {feat:20s}: {imp:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
