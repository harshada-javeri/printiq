"""
SHAP-based model explainability.

Provides TreeExplainer for RandomForest models and per-job explanations.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import shap

from .config import (
    PREPROCESSOR_FILE,
    FAILURE_MODEL_FILE,
    QUALITY_MODEL_FILE,
    FAILURE_EXPLAINER_FILE,
    QUALITY_EXPLAINER_FILE,
    FEATURE_SET,
)

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Wrapper for SHAP-based model explanations."""
    
    def __init__(self, model_path: Path, explainer_path: Path):
        """
        Initialize explainer.
        
        Args:
            model_path: Path to trained model
            explainer_path: Path to save/load explainer
        """
        self.model_path = model_path
        self.explainer_path = explainer_path
        self.model = None
        self.explainer = None
        self.background_data = None
    
    def load_model(self):
        """Load trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        logger.info(f"Loaded model from {self.model_path}")
    
    def create_explainer(self, X_background: np.ndarray, max_samples: int = 100):
        """
        Create SHAP TreeExplainer using background data.
        
        Args:
            X_background: Background data for SHAP (e.g., training data)
            max_samples: Maximum samples to use (for efficiency)
        """
        if self.model is None:
            self.load_model()
        
        # Sample background data if needed
        if len(X_background) > max_samples:
            background_indices = np.random.choice(
                len(X_background),
                max_samples,
                replace=False
            )
            X_background = X_background[background_indices]
        
        self.background_data = X_background
        self.explainer = shap.TreeExplainer(self.model, data=X_background)
        logger.info(f"Created TreeExplainer with {len(X_background)} background samples")
    
    def save_explainer(self):
        """Save explainer to disk."""
        if self.explainer is None:
            raise ValueError("No explainer to save. Create one first.")
        
        self.explainer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.explainer, self.explainer_path)
        logger.info(f"Saved explainer to {self.explainer_path}")
    
    def load_explainer(self):
        """Load explainer from disk."""
        if not self.explainer_path.exists():
            raise FileNotFoundError(f"Explainer not found: {self.explainer_path}")
        
        self.explainer = joblib.load(self.explainer_path)
        logger.info(f"Loaded explainer from {self.explainer_path}")
    
    def explain(self, X: np.ndarray) -> dict:
        """
        Generate SHAP explanation for a sample.
        
        Args:
            X: Input features (single sample or batch)
            
        Returns:
            Dict with SHAP values and metadata
        """
        if self.explainer is None:
            self.load_explainer()
        
        # Ensure input is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle multi-class/multi-output cases
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, use class 1
        
        # Return explanation for first sample
        explanation = {
            "shap_values": {feat: float(val) for feat, val in zip(FEATURE_SET, shap_values[0])},
            "base_value": float(self.explainer.expected_value),
        }
        
        return explanation


class FailureExplainer(ModelExplainer):
    """Explainer for failure prediction model."""
    
    def __init__(self):
        super().__init__(FAILURE_MODEL_FILE, FAILURE_EXPLAINER_FILE)
    
    def explain_with_prediction(self, X: np.ndarray, preprocessor) -> dict:
        """
        Generate failure explanation with prediction.
        
        Args:
            X: Raw input features (before preprocessing)
            preprocessor: Feature preprocessor
            
        Returns:
            Dict with prediction and explanation
        """
        # Transform features
        X_processed = preprocessor.transform(X.reshape(1, -1))
        
        # Get prediction
        y_pred_proba = self.model.predict_proba(X_processed)[0]
        failure_prob = float(y_pred_proba[1])
        predicted_class = int(np.argmax(y_pred_proba))
        
        # Get explanation
        explanation = self.explain(X_processed)
        
        return {
            "failure_probability": failure_prob,
            "predicted_class": predicted_class,
            **explanation,
        }


class QualityExplainer(ModelExplainer):
    """Explainer for quality prediction model."""
    
    def __init__(self):
        super().__init__(QUALITY_MODEL_FILE, QUALITY_EXPLAINER_FILE)
    
    def explain_with_prediction(self, X: np.ndarray, preprocessor) -> dict:
        """
        Generate quality explanation with prediction.
        
        Args:
            X: Raw input features (before preprocessing)
            preprocessor: Feature preprocessor
            
        Returns:
            Dict with prediction and explanation
        """
        # Transform features
        X_processed = preprocessor.transform(X.reshape(1, -1))
        
        # Get prediction
        quality_score = float(self.model.predict(X_processed)[0])
        
        # Get explanation
        explanation = self.explain(X_processed)
        
        return {
            "quality_score": quality_score,
            **explanation,
        }


def initialize_explainers(X_background: np.ndarray):
    """
    Initialize both failure and quality explainers.
    
    Args:
        X_background: Background data for SHAP (preprocessed)
    """
    logger.info("Initializing SHAP explainers...")
    
    failure_explainer = FailureExplainer()
    failure_explainer.load_model()
    failure_explainer.create_explainer(X_background)
    failure_explainer.save_explainer()
    
    quality_explainer = QualityExplainer()
    quality_explainer.load_model()
    quality_explainer.create_explainer(X_background)
    quality_explainer.save_explainer()
    
    logger.info("✓ Explainers ready for use")
