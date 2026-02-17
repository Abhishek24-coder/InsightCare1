# app/api/diagnosis.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List
import uuid
from app.database import get_db
from app.schemas.diagnosis_schema import (
    DiagnosisRequest,
    DiagnosisOut,
    DiagnosisHistoryResponse,
    DiagnosisHistoryItem,
    PredictionOut,
)
from app.services.diagnosis_service import (
    create_diagnosis,
    get_user_diagnosis_history,
    get_diagnosis_by_id,
)
from app.utils.dependencies import get_current_user
from app.models.user import User

router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])
# Create a separate router for frontend compatibility (without /diagnosis prefix)
diagnose_router = APIRouter(tags=["Diagnosis"])


@router.post("/analyze", response_model=DiagnosisOut)
def analyze_symptoms(
    request_data: DiagnosisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Analyze symptoms and get AI-powered disease predictions.

    - **symptoms**: List of symptoms (1-20 items, 2-100 chars each)
    - **severity**: Optional overall severity (mild/moderate/severe)
    - **duration**: Optional duration description

    Returns top 3-5 disease predictions with confidence scores and recommendations.
    Requires valid JWT token.
    """
    try:
        diagnosis = create_diagnosis(db, current_user.id, request_data)

        # Convert to response model
        predictions = [PredictionOut(**pred) for pred in diagnosis.predictions]

        return DiagnosisOut(
            diagnosis_id=diagnosis.id,
            timestamp=diagnosis.created_at,
            symptoms_analyzed=diagnosis.symptoms,
            predictions=predictions,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagnosis analysis failed: {str(e)}",
        )


# Add /diagnose endpoint to match frontend expectations
@router.post("/diagnose", response_model=DiagnosisOut)
def diagnose_symptoms(
    request_data: DiagnosisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Diagnose symptoms endpoint (alternative to /analyze for frontend compatibility).
    """
    return analyze_symptoms(request_data, current_user, db)


@router.get("/history", response_model=DiagnosisHistoryResponse)
def get_history(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=50, description="Items per page (max 50)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get paginated diagnosis history for current user.

    - **page**: Page number (default 1)
    - **limit**: Items per page (default 10, max 50)

    Returns list of past diagnoses with timestamps and top predictions.
    Requires valid JWT token.
    """
    try:
        diagnoses, total = get_user_diagnosis_history(db, current_user.id, page, limit)

        # Convert to history items
        results = []
        for diag in diagnoses:
            top_pred = (
                diag.predictions[0]
                if diag.predictions
                else {"disease": "Unknown", "confidence": 0.0}
            )
            results.append(
                DiagnosisHistoryItem(
                    diagnosis_id=diag.id,
                    timestamp=diag.created_at,
                    symptoms=diag.symptoms,
                    top_prediction={
                        "disease": top_pred["disease"],
                        "confidence": top_pred["confidence"],
                    },
                )
            )

        return DiagnosisHistoryResponse(
            total=total, page=page, limit=limit, results=results
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}",
        )


@router.get("/{diagnosis_id}", response_model=DiagnosisOut)
def get_diagnosis_detail(
    diagnosis_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get detailed diagnosis by ID.

    - **diagnosis_id**: UUID of the diagnosis

    Returns full diagnosis details including all predictions.
    Only accessible by the user who created it. Raises 404 if not found.
    Requires valid JWT token.
    """
    diagnosis = get_diagnosis_by_id(db, diagnosis_id, current_user.id)

    predictions = [PredictionOut(**pred) for pred in diagnosis.predictions]

    return DiagnosisOut(
        diagnosis_id=diagnosis.id,
        timestamp=diagnosis.created_at,
        symptoms_analyzed=diagnosis.symptoms,
        predictions=predictions,
    )


# Frontend-compatible endpoints (without /diagnosis prefix)
@diagnose_router.post("/diagnose", response_model=DiagnosisOut)
def diagnose_symptoms_frontend(
    request_data: DiagnosisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Diagnose symptoms endpoint for frontend compatibility.
    Maps to /api/diagnose
    """
    return analyze_symptoms(request_data, current_user, db)


@diagnose_router.get("/history", response_model=DiagnosisHistoryResponse)
def get_history_frontend(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=50, description="Items per page (max 50)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get diagnosis history endpoint for frontend compatibility.
    Maps to /api/history
    """
    return get_history(page, limit, current_user, db)
