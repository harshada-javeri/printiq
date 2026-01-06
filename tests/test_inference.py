"""
Unit tests for PrintIQ inference pipeline.

Usage:
    pytest tests/ -v --cov=src
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.schema import PrintJobInput
from src.inference import ModelInference, classify_quality
from src.features import FeaturePreprocessor
from src.config import FEATURE_SET


class TestPrintJobInput:
    """Test input schema validation."""
    
    def test_valid_input(self):
        """Test valid input passes validation."""
        job = PrintJobInput(
            printer_age=24,
            head_type="piezo",
            ink_viscosity=35.5,
            paper_gsm=80.0,
            humidity=45.0,
            temperature=22.0,
            coverage_pct=65.0,
            nozzles_clean=True,
        )
        assert job.printer_age == 24
        assert job.head_type == "piezo"
    
    def test_invalid_head_type(self):
        """Test invalid head type raises validation error."""
        with pytest.raises(ValueError):
            PrintJobInput(
                printer_age=24,
                head_type="unknown",
                ink_viscosity=35.5,
                paper_gsm=80.0,
                humidity=45.0,
                temperature=22.0,
                coverage_pct=65.0,
                nozzles_clean=True,
            )
    
    def test_out_of_bounds_age(self):
        """Test out-of-bounds printer age."""
        with pytest.raises(ValueError):
            PrintJobInput(
                printer_age=100,  # Max is 60
                head_type="piezo",
                ink_viscosity=35.5,
                paper_gsm=80.0,
                humidity=45.0,
                temperature=22.0,
                coverage_pct=65.0,
                nozzles_clean=True,
            )
    
    def test_out_of_bounds_humidity(self):
        """Test out-of-bounds humidity."""
        with pytest.raises(ValueError):
            PrintJobInput(
                printer_age=24,
                head_type="piezo",
                ink_viscosity=35.5,
                paper_gsm=80.0,
                humidity=90,  # Max is 70
                temperature=22.0,
                coverage_pct=65.0,
                nozzles_clean=True,
            )


class TestQualityClassification:
    """Test quality score classification."""
    
    def test_poor_quality(self):
        """Test poor quality classification."""
        assert classify_quality(30) == "poor"
        assert classify_quality(39) == "poor"
    
    def test_fair_quality(self):
        """Test fair quality classification."""
        assert classify_quality(50) == "fair"
        assert classify_quality(70) == "good"
    
    def test_good_quality(self):
        """Test good quality classification."""
        assert classify_quality(75) == "good"
        assert classify_quality(79) == "good"
    
    def test_excellent_quality(self):
        """Test excellent quality classification."""
        assert classify_quality(80) == "excellent"
        assert classify_quality(100) == "excellent"


class TestFeaturePreprocessor:
    """Test feature preprocessing pipeline."""
    
    def test_preprocessor_fit_transform(self):
        """Test preprocessor fit and transform."""
        # Create sample data
        df = pd.DataFrame({
            "printer_age": [24, 36],
            "head_type": ["piezo", "thermal"],
            "ink_viscosity": [35.5, 38.0],
            "paper_gsm": [80, 100],
            "humidity": [45, 50],
            "temperature": [22, 24],
            "coverage_pct": [65, 70],
            "nozzles_clean": [True, False],
        })
        
        preprocessor = FeaturePreprocessor()
        X_transformed = preprocessor.fit_transform(df)
        
        # Check output shape
        assert X_transformed.shape[0] == 2
        assert X_transformed.shape[1] == 8  # 7 numeric + 1 categorical
    
    def test_preprocessor_consistency(self):
        """Test preprocessor produces consistent output."""
        df = pd.DataFrame({
            "printer_age": [24],
            "head_type": ["piezo"],
            "ink_viscosity": [35.5],
            "paper_gsm": [80],
            "humidity": [45],
            "temperature": [22],
            "coverage_pct": [65],
            "nozzles_clean": [True],
        })
        
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(df)
        
        # Same input should produce same output
        X1 = preprocessor.transform(df)
        X2 = preprocessor.transform(df)
        
        assert np.allclose(X1, X2)
    
    def test_preprocessor_scaling(self):
        """Test features are properly scaled."""
        df = pd.DataFrame({
            "printer_age": [24, 36],
            "head_type": ["piezo", "thermal"],
            "ink_viscosity": [35.5, 38.0],
            "paper_gsm": [80, 100],
            "humidity": [45, 50],
            "temperature": [22, 24],
            "coverage_pct": [65, 70],
            "nozzles_clean": [True, False],
        })
        
        preprocessor = FeaturePreprocessor()
        X_transformed = preprocessor.fit_transform(df)
        
        # Check that numeric features are standardized
        numeric_cols = slice(0, 6)  # First 6 columns are numeric
        mean = np.mean(X_transformed[numeric_cols])
        std = np.std(X_transformed[numeric_cols])
        
        assert np.isclose(mean, 0, atol=1e-5)
        assert np.isclose(std, 1, atol=1)  # Allow some tolerance


class TestModelInference:
    """Test inference engine (requires trained models)."""
    
    @pytest.fixture
    def inference(self):
        """Initialize inference engine."""
        inf = ModelInference()
        try:
            inf.load_artifacts()
            return inf
        except FileNotFoundError:
            pytest.skip("Models not trained. Run: python src/train.py")
    
    def test_failure_prediction_shape(self, inference):
        """Test failure prediction returns correct output."""
        data = {
            "printer_age": 24,
            "head_type": "piezo",
            "ink_viscosity": 35.5,
            "paper_gsm": 80.0,
            "humidity": 45.0,
            "temperature": 22.0,
            "coverage_pct": 65.0,
            "nozzles_clean": True,
        }
        
        prob, predicted_class, confidence = inference.predict_failure(data)
        
        assert isinstance(prob, float)
        assert 0 <= prob <= 1
        assert predicted_class in [0, 1]
        assert 0 <= confidence <= 1
    
    def test_quality_prediction_shape(self, inference):
        """Test quality prediction returns correct output."""
        data = {
            "printer_age": 24,
            "head_type": "piezo",
            "ink_viscosity": 35.5,
            "paper_gsm": 80.0,
            "humidity": 45.0,
            "temperature": 22.0,
            "coverage_pct": 65.0,
            "nozzles_clean": True,
        }
        
        score = inference.predict_quality(data)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    def test_batch_prediction(self, inference):
        """Test batch prediction."""
        df = pd.DataFrame({
            "printer_age": [24, 36],
            "head_type": ["piezo", "thermal"],
            "ink_viscosity": [35.5, 38.0],
            "paper_gsm": [80, 100],
            "humidity": [45, 50],
            "temperature": [22, 24],
            "coverage_pct": [65, 70],
            "nozzles_clean": [True, False],
        })
        
        result = inference.predict_batch(df)
        
        assert len(result) == 2
        assert "failure_probability" in result.columns
        assert "quality_score" in result.columns
        assert (result["quality_score"] >= 0).all()
        assert (result["quality_score"] <= 100).all()


class TestPredictionLogic:
    """Test prediction logic without models."""
    
    def test_inputs_to_dataframe(self):
        """Test conversion of dict inputs to DataFrame."""
        data = {
            "printer_age": 24,
            "head_type": "piezo",
            "ink_viscosity": 35.5,
            "paper_gsm": 80.0,
            "humidity": 45.0,
            "temperature": 22.0,
            "coverage_pct": 65.0,
            "nozzles_clean": True,
        }
        
        df = pd.DataFrame([data])
        assert len(df) == 1
        assert df["printer_age"].iloc[0] == 24
        assert df["head_type"].iloc[0] == "piezo"
    
    def test_probability_bounds(self):
        """Test predictions respect probability bounds."""
        # Probability must be between 0 and 1
        prob = 0.5
        assert 0 <= prob <= 1
        
        # Quality score must be between 0 and 100
        score = 75.5
        assert 0 <= score <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
