"""
API routes for print job prediction and explanation.
"""

import logging
from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd

from ..src.schema import PrintJobInput, FailurePredictionResponse, QualityPredictionResponse, SHAPExplanation
from ..src.inference import ModelInference, classify_quality
from ..src.explain import FailureExplainer, QualityExplainer
from .deps import get_model_container

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["predictions"])

# Initialize inference engine
inference_engine = ModelInference()


@router.post("/predict/failure", response_model=FailurePredictionResponse)
def predict_failure(job: PrintJobInput):
    """
    Predict failure probability for a print job.
    
    Args:
        job: Print job features
        
    Returns:
        Failure prediction with probability and confidence
        
    Example:
        POST /api/v1/predict/failure
        {
            "printer_age": 24,
            "head_type": "piezo",
            "ink_viscosity": 35.5,
            "paper_gsm": 80.0,
            "humidity": 45.0,
            "temperature": 22.0,
            "coverage_pct": 65.0,
            "nozzles_clean": true
        }
    """
    try:
        # Convert input to dict
        job_data = job.model_dump()
        
        # Make prediction
        failure_prob, predicted_class, confidence = inference_engine.predict_failure(job_data)
        
        return FailurePredictionResponse(
            failure_probability=failure_prob,
            predicted_class=predicted_class,
            confidence=confidence,
        )
    
    except Exception as e:
        logger.error(f"Failure prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/quality", response_model=QualityPredictionResponse)
def predict_quality(job: PrintJobInput):
    """
    Predict quality score for a print job.
    
    Args:
        job: Print job features
        
    Returns:
        Quality score (0-100) and category
    """
    try:
        # Convert input to dict
        job_data = job.model_dump()
        
        # Make prediction
        quality_score = inference_engine.predict_quality(job_data)
        
        # Classify quality
        quality_category = classify_quality(quality_score)
        
        return QualityPredictionResponse(
            quality_score=quality_score,
            quality_category=quality_category,
        )
    
    except Exception as e:
        logger.error(f"Quality prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/explain/failure", response_model=SHAPExplanation)
def explain_failure(job: PrintJobInput):
    """
    Generate SHAP-based explanation for failure prediction.
    
    Returns failure probability with per-feature contribution.
    """
    try:
        # Prepare input
        job_dict = job.model_dump()
        df = pd.DataFrame([job_dict])
        X = df[["printer_age", "head_type", "ink_viscosity", "paper_gsm", 
                "humidity", "temperature", "coverage_pct", "nozzles_clean"]].values
        
        # Initialize explainer
        explainer = FailureExplainer()
        explainer.load_model()
        
        # Get preprocessing
        preprocessor = inference_engine.preprocessor
        if preprocessor is None:
            inference_engine.load_artifacts()
            preprocessor = inference_engine.preprocessor
        
        X_processed = preprocessor.transform(df)
        
        # Get prediction
        y_pred = explainer.model.predict_proba(X_processed)[0]
        failure_prob = float(y_pred[1])
        predicted_class = int(np.argmax(y_pred))
        
        # Get explanation
        try:
            explainer.load_explainer()
            shap_vals = explainer.explainer.shap_values(X_processed)
            
            # Handle different output formats
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            
            feature_names = ["printer_age", "head_type", "ink_viscosity", "paper_gsm",
                           "humidity", "temperature", "coverage_pct", "nozzles_clean"]
            
            shap_dict = {feat: float(val) for feat, val in zip(feature_names, shap_vals[0])}
            base_value = float(explainer.explainer.expected_value)
        except:
            # Fallback if explainer not available
            shap_dict = {feat: 0.0 for feat in feature_names}
            base_value = failure_prob
        
        return SHAPExplanation(
            failure_probability=failure_prob,
            predicted_class=predicted_class,
            shap_values=shap_dict,
            base_value=base_value,
        )
    
    except Exception as e:
        logger.error(f"Explanation generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.post("/explain/quality", response_model=SHAPExplanation)
def explain_quality(job: PrintJobInput):
    """
    Generate SHAP-based explanation for quality prediction.
    
    Returns quality score with per-feature contribution.
    """
    try:
        # Prepare input
        job_dict = job.model_dump()
        df = pd.DataFrame([job_dict])
        X = df[["printer_age", "head_type", "ink_viscosity", "paper_gsm",
                "humidity", "temperature", "coverage_pct", "nozzles_clean"]].values
        
        # Initialize explainer
        explainer = QualityExplainer()
        explainer.load_model()
        
        # Get preprocessing
        preprocessor = inference_engine.preprocessor
        if preprocessor is None:
            inference_engine.load_artifacts()
            preprocessor = inference_engine.preprocessor
        
        X_processed = preprocessor.transform(df)
        
        # Get prediction
        quality_score = float(explainer.model.predict(X_processed)[0])
        
        # Get explanation
        try:
            explainer.load_explainer()
            shap_vals = explainer.explainer.shap_values(X_processed)
            
            # Handle different output formats
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            
            feature_names = ["printer_age", "head_type", "ink_viscosity", "paper_gsm",
                           "humidity", "temperature", "coverage_pct", "nozzles_clean"]
            
            shap_dict = {feat: float(val) for feat, val in zip(feature_names, shap_vals[0])}
            base_value = float(explainer.explainer.expected_value)
        except:
            # Fallback if explainer not available
            shap_dict = {feat: 0.0 for feat in feature_names}
            base_value = quality_score
        
        return SHAPExplanation(
            failure_probability=quality_score / 100.0,  # Normalize to 0-1
            predicted_class=1 if quality_score >= 70 else 0,  # Map to class
            shap_values=shap_dict,
            base_value=base_value,
        )
    
    except Exception as e:
        logger.error(f"Quality explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
