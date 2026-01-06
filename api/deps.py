"""
Dependency injection and shared utilities for FastAPI.
"""

import logging
from functools import lru_cache
from pathlib import Path

from ..src.config import PREPROCESSOR_FILE, FAILURE_MODEL_FILE, QUALITY_MODEL_FILE

logger = logging.getLogger(__name__)


class ModelContainer:
    """Singleton container for loaded models."""
    
    _instance = None
    _preprocessor = None
    _failure_model = None
    _quality_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_models()
        return cls._instance
    
    def _load_models(self):
        """Load all required models and preprocessor."""
        import joblib
        
        if not all([
            PREPROCESSOR_FILE.exists(),
            FAILURE_MODEL_FILE.exists(),
            QUALITY_MODEL_FILE.exists(),
        ]):
            raise RuntimeError(
                "Model artifacts not found. Run: python src/train.py\n"
                f"  - {PREPROCESSOR_FILE}\n"
                f"  - {FAILURE_MODEL_FILE}\n"
                f"  - {QUALITY_MODEL_FILE}"
            )
        
        try:
            self._preprocessor = joblib.load(PREPROCESSOR_FILE)
            self._failure_model = joblib.load(FAILURE_MODEL_FILE)
            self._quality_model = joblib.load(QUALITY_MODEL_FILE)
            logger.info("✓ Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    @property
    def preprocessor(self):
        """Get preprocessor."""
        return self._preprocessor
    
    @property
    def failure_model(self):
        """Get failure prediction model."""
        return self._failure_model
    
    @property
    def quality_model(self):
        """Get quality prediction model."""
        return self._quality_model


@lru_cache(maxsize=1)
def get_model_container() -> ModelContainer:
    """Get singleton model container (dependency injection)."""
    return ModelContainer()
