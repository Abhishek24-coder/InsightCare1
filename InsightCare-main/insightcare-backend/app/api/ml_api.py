"""
ML API Endpoint - /api/ml/diagnose
Direct ML prediction endpoint for frontend
"""

from fastapi import APIRouter, HTTPException, status
from typing import List
from pydantic import BaseModel, Field
from app.services.ml_service import get_ml_diagnosis, is_ml_available, ml_service

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


class MLDiagnosisRequest(BaseModel):
    """Request model for ML diagnosis"""
    symptoms: List[str] = Field(
        ...,
        min_items=1,
        max_items=20,
        description="List of symptoms (1-20 items)",
        example=["fever", "cough", "fatigue", "headache"]
    )


class MLPrediction(BaseModel):
    """Single ML prediction"""
    disease: str
    confidence: float
    severity: str
    description: str
    recommendations: List[str]
    model_used: str
    valid_symptoms: List[str]
    invalid_symptoms: List[str]


class MLDiagnosisResponse(BaseModel):
    """Response model for ML diagnosis"""
    success: bool
    predictions: List[MLPrediction]
    total_predictions: int
    ml_available: bool
    message: str


class MLHealthResponse(BaseModel):
    """ML system health check"""
    ml_available: bool
    total_symptoms: int
    total_diseases: int
    models_loaded: List[str]


@router.post("/diagnose", response_model=MLDiagnosisResponse)
def diagnose_with_ml(request: MLDiagnosisRequest):
    """
    Get disease diagnosis using ML models (Random Forest + XGBoost)
    
    - **symptoms**: List of symptom names (e.g., ["fever", "cough", "headache"])
    - Returns predictions from trained ML models with confidence scores
    - No authentication required for ML prediction
    
    Example:
    ```json
    {
        "symptoms": ["fever", "cough", "fatigue", "headache"]
    }
    ```
    """
    try:
        # Get ML predictions
        predictions = get_ml_diagnosis(request.symptoms)
        
        return MLDiagnosisResponse(
            success=True,
            predictions=predictions,
            total_predictions=len(predictions),
            ml_available=is_ml_available(),
            message="ML prediction successful" if is_ml_available() else "Using fallback diagnosis"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ML diagnosis failed: {str(e)}"
        )


@router.get("/health", response_model=MLHealthResponse)
def ml_health_check():
    """
    Check ML system health and availability
    
    Returns information about loaded models and available data
    """
    if not is_ml_available():
        return MLHealthResponse(
            ml_available=False,
            total_symptoms=0,
            total_diseases=0,
            models_loaded=[]
        )
    
    return MLHealthResponse(
        ml_available=True,
        total_symptoms=len(ml_service.get_available_symptoms()),
        total_diseases=len(ml_service.get_available_diseases()),
        models_loaded=["Random Forest (100 trees)", "XGBoost (100 estimators)"]
    )


@router.get("/symptoms", response_model=List[str])
def get_available_symptoms():
    """
    Get list of all available symptoms that the ML model recognizes
    
    Returns 131 symptom names that can be used for diagnosis
    """
    if not is_ml_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not available"
        )
    
    return ml_service.get_available_symptoms()


@router.get("/diseases", response_model=List[str])
def get_available_diseases():
    """
    Get list of all diseases that the ML model can predict
    
    Returns 41 disease names
    """
    if not is_ml_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not available"
        )
    
    return ml_service.get_available_diseases()
