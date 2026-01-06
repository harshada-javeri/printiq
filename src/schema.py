"""
Pydantic schemas for input validation and type safety.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class PrintJobInput(BaseModel):
    """Input schema for print job prediction requests."""
    
    printer_age: float = Field(..., ge=1, le=60, description="Printer age in months")
    head_type: str = Field(..., description="Printer head type: 'piezo' or 'thermal'")
    ink_viscosity: float = Field(..., ge=25, le=45, description="Ink viscosity in centiPoise (cP)")
    paper_gsm: float = Field(..., ge=70, le=300, description="Paper weight in grams/m²")
    humidity: float = Field(..., ge=30, le=70, description="Humidity percentage")
    temperature: float = Field(..., ge=18, le=28, description="Temperature in Celsius")
    coverage_pct: float = Field(..., ge=10, le=95, description="Page coverage percentage")
    nozzles_clean: bool = Field(..., description="Whether nozzles are clean")
    
    @field_validator('head_type')
    @classmethod
    def validate_head_type(cls, v: str) -> str:
        """Validate head type is one of allowed values."""
        if v not in ["piezo", "thermal"]:
            raise ValueError("head_type must be 'piezo' or 'thermal'")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "printer_age": 24,
                "head_type": "piezo",
                "ink_viscosity": 35.5,
                "paper_gsm": 80.0,
                "humidity": 45.0,
                "temperature": 22.0,
                "coverage_pct": 65.0,
                "nozzles_clean": True,
            }
        }


class FailurePredictionResponse(BaseModel):
    """Response schema for failure prediction."""
    
    failure_probability: float = Field(..., ge=0, le=1, description="Probability of failure (0-1)")
    predicted_class: int = Field(..., description="Predicted class: 0=success, 1=failure")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in prediction")


class QualityPredictionResponse(BaseModel):
    """Response schema for quality prediction."""
    
    quality_score: float = Field(..., ge=0, le=100, description="Predicted quality score (0-100)")
    quality_category: str = Field(..., description="Quality category: poor/fair/good/excellent")


class SHAPExplanation(BaseModel):
    """Response schema for SHAP explanations."""
    
    failure_probability: float = Field(..., description="Failure probability")
    predicted_class: int = Field(..., description="Predicted failure class")
    shap_values: dict = Field(..., description="SHAP values for each feature")
    base_value: float = Field(..., description="Model base value (average prediction)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "failure_probability": 0.12,
                "predicted_class": 0,
                "shap_values": {
                    "printer_age": -0.03,
                    "head_type": 0.05,
                    "ink_viscosity": 0.02,
                    "paper_gsm": -0.01,
                    "humidity": 0.04,
                    "temperature": -0.02,
                    "coverage_pct": 0.06,
                    "nozzles_clean": 0.01,
                },
                "base_value": 0.10,
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
