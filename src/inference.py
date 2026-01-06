"""
Inference pipeline for making predictions with trained models.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Union, Tuple

from .config import (
    PREPROCESSOR_FILE,
    FAILURE_MODEL_FILE,
    QUALITY_MODEL_FILE,
    FEATURE_SET,
)


class ModelInference:
    """Unified inference interface for failure and quality models."""
    
    def __init__(self):
        """Initialize inference engine."""
        self.preprocessor = None
        self.failure_model = None
        self.quality_model = None
        self.loaded = False
    
    def load_artifacts(self):
        """Load all trained models and preprocessor from disk."""
        if not all([
            PREPROCESSOR_FILE.exists(),
            FAILURE_MODEL_FILE.exists(),
            QUALITY_MODEL_FILE.exists(),
        ]):
            raise FileNotFoundError(
                "Model artifacts not found. Run: python src/train.py"
            )
        
        self.preprocessor = joblib.load(PREPROCESSOR_FILE)
        self.failure_model = joblib.load(FAILURE_MODEL_FILE)
        self.quality_model = joblib.load(QUALITY_MODEL_FILE)
        self.loaded = True
    
    def _prepare_input(self, data: Union[dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Convert input to numpy array with proper feature ordering.
        
        Args:
            data: Input data (dict, DataFrame, or array)
            
        Returns:
            Numpy array with correct shape for models
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                df = pd.DataFrame([data], columns=FEATURE_SET)
            else:
                df = pd.DataFrame(data, columns=FEATURE_SET)
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
        
        # Ensure correct columns and order
        df = df[FEATURE_SET]
        
        # Handle categorical encoding
        if "head_type" in df.columns:
            if isinstance(df["head_type"].iloc[0], str):
                # Already strings, preprocessor will handle
                pass
        
        return df
    
    def predict_failure(self, data: Union[dict, pd.DataFrame]) -> Tuple[float, int, float]:
        """
        Predict failure probability.
        
        Args:
            data: Input features as dict or DataFrame
            
        Returns:
            Tuple of (failure_probability, predicted_class, confidence)
        """
        if not self.loaded:
            self.load_artifacts()
        
        df = self._prepare_input(data)
        X_processed = self.preprocessor.transform(df)
        
        # Get predictions
        y_pred = self.failure_model.predict(X_processed)[0]
        y_proba = self.failure_model.predict_proba(X_processed)[0]
        
        failure_prob = float(y_proba[1])
        confidence = float(np.max(y_proba))
        
        return failure_prob, int(y_pred), confidence
    
    def predict_quality(self, data: Union[dict, pd.DataFrame]) -> float:
        """
        Predict quality score.
        
        Args:
            data: Input features as dict or DataFrame
            
        Returns:
            Quality score (0-100)
        """
        if not self.loaded:
            self.load_artifacts()
        
        df = self._prepare_input(data)
        X_processed = self.preprocessor.transform(df)
        
        quality_score = float(self.quality_model.predict(X_processed)[0])
        
        return np.clip(quality_score, 0, 100)
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on multiple samples.
        
        Args:
            data: DataFrame with features
            
        Returns:
            DataFrame with failure_prob, quality_score columns added
        """
        if not self.loaded:
            self.load_artifacts()
        
        df = self._prepare_input(data).copy()
        X_processed = self.preprocessor.transform(df)
        
        # Failure predictions
        df["failure_probability"] = self.failure_model.predict_proba(X_processed)[:, 1]
        df["predicted_failure"] = self.failure_model.predict(X_processed)
        
        # Quality predictions
        df["quality_score"] = np.clip(
            self.quality_model.predict(X_processed),
            0, 100
        )
        
        return df


def classify_quality(score: float) -> str:
    """
    Classify quality score into categories.
    
    Args:
        score: Quality score 0-100
        
    Returns:
        Category label: 'poor', 'fair', 'good', or 'excellent'
    """
    if score < 40:
        return "poor"
    elif score < 60:
        return "fair"
    elif score < 80:
        return "good"
    else:
        return "excellent"
