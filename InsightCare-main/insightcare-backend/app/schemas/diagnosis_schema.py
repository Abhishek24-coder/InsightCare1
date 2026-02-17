# app/schemas/diagnosis_schema.py
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
from typing import List, Optional
import uuid


class PredictionOut(BaseModel):
    """Schema for a single disease prediction."""

    disease: str = Field(..., description="Disease name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    severity: str = Field(..., description="Severity level (mild/moderate/severe)")
    description: str = Field(..., description="Brief disease description")
    recommendations: List[str] = Field(..., description="List of recommendations")


class DiagnosisRequest(BaseModel):
    """Schema for diagnosis analysis request."""

    symptoms: List[str] = Field(
        ..., min_length=1, max_length=20, description="List of symptoms"
    )
    severity: Optional[str] = Field(
        None, description="Overall severity (mild/moderate/severe)"
    )
    duration: Optional[str] = Field(
        None, max_length=100, description="Duration of symptoms"
    )

    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, v):
        for symptom in v:
            if len(symptom) < 2 or len(symptom) > 100:
                raise ValueError("Each symptom must be 2-100 characters")
        return [s.strip().lower() for s in v]

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v):
        if v and v.lower() not in ["mild", "moderate", "severe"]:
            raise ValueError("Severity must be mild, moderate, or severe")
        return v.lower() if v else None


class DiagnosisOut(BaseModel):
    """Schema for diagnosis analysis response."""

    model_config = ConfigDict(from_attributes=True)

    diagnosis_id: uuid.UUID
    timestamp: datetime
    symptoms_analyzed: List[str]
    predictions: List[PredictionOut]
    disclaimer: str = (
        "This is an AI-generated prediction. Please consult a healthcare professional for accurate diagnosis."
    )


class DiagnosisHistoryItem(BaseModel):
    """Schema for a single diagnosis history item."""

    model_config = ConfigDict(from_attributes=True)

    diagnosis_id: uuid.UUID
    timestamp: datetime
    symptoms: List[str]
    top_prediction: dict  # {"disease": str, "confidence": float}


class DiagnosisHistoryResponse(BaseModel):
    """Schema for diagnosis history response."""

    total: int
    page: int
    limit: int
    results: List[DiagnosisHistoryItem]
