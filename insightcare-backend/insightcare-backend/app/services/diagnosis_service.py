# app/services/diagnosis_service.py
from typing import List, Dict
from sqlalchemy.orm import Session
import uuid
from datetime import datetime
from app.models.diagnosis import Diagnosis
from app.schemas.diagnosis_schema import DiagnosisRequest, PredictionOut


def mock_ai_diagnosis(symptoms: List[str]) -> List[Dict]:
    """
    Mock AI diagnosis function using rule-based matching.
    This is a Phase 2 placeholder before real AI integration.

    Args:
        symptoms: List of symptom strings (lowercased)

    Returns:
        List of prediction dictionaries
    """
    predictions = []

    # Rule 1: Cold/Flu symptoms
    if any(
        s in symptoms
        for s in ["fever", "cough", "headache", "sore throat", "runny nose"]
    ):
        predictions.append(
            {
                "disease": "Common Cold",
                "confidence": 0.85,
                "severity": "mild",
                "description": "Viral infection affecting the upper respiratory tract",
                "recommendations": [
                    "Get plenty of rest",
                    "Drink fluids to stay hydrated",
                    "Use over-the-counter cold medications",
                    "Gargle with salt water for sore throat",
                ],
            }
        )

    # Rule 2: Flu symptoms
    if any(s in symptoms for s in ["high fever", "body aches", "fatigue", "chills"]):
        predictions.append(
            {
                "disease": "Influenza (Flu)",
                "confidence": 0.78,
                "severity": "moderate",
                "description": "Contagious respiratory illness caused by influenza viruses",
                "recommendations": [
                    "Rest and sleep as much as possible",
                    "Drink plenty of fluids",
                    "Consider antiviral medications if within 48 hours",
                    "Isolate from others to prevent spread",
                ],
            }
        )

    # Rule 3: Allergies
    if any(
        s in symptoms for s in ["sneezing", "itchy eyes", "watery eyes", "congestion"]
    ):
        predictions.append(
            {
                "disease": "Seasonal Allergies",
                "confidence": 0.72,
                "severity": "mild",
                "description": "Allergic reaction to airborne substances like pollen",
                "recommendations": [
                    "Use antihistamine medications",
                    "Avoid known allergens",
                    "Keep windows closed during high pollen days",
                    "Consider allergy testing",
                ],
            }
        )

    # Rule 4: Migraine
    if any(
        s in symptoms
        for s in ["severe headache", "nausea", "sensitivity to light", "dizziness"]
    ):
        predictions.append(
            {
                "disease": "Migraine",
                "confidence": 0.80,
                "severity": "moderate",
                "description": "Intense headache often accompanied by nausea and light sensitivity",
                "recommendations": [
                    "Rest in a quiet, dark room",
                    "Apply cold compress to head",
                    "Take migraine-specific medication",
                    "Identify and avoid triggers",
                ],
            }
        )

    # Rule 5: Gastroenteritis
    if any(
        s in symptoms
        for s in ["nausea", "vomiting", "diarrhea", "stomach pain", "cramping"]
    ):
        predictions.append(
            {
                "disease": "Gastroenteritis (Stomach Flu)",
                "confidence": 0.76,
                "severity": "moderate",
                "description": "Inflammation of the digestive tract causing stomach upset",
                "recommendations": [
                    "Stay hydrated with clear fluids",
                    "Follow BRAT diet (bananas, rice, applesauce, toast)",
                    "Avoid dairy and fatty foods",
                    "Rest and allow recovery time",
                ],
            }
        )

    # Default: Unknown condition
    if not predictions:
        predictions.append(
            {
                "disease": "Unspecified Condition",
                "confidence": 0.45,
                "severity": "unknown",
                "description": "Symptoms do not match common patterns in our database",
                "recommendations": [
                    "Consult a healthcare professional",
                    "Monitor symptoms closely",
                    "Keep a symptom diary",
                    "Seek immediate care if symptoms worsen",
                ],
            }
        )

    # Sort by confidence and return top 5
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    return predictions[:5]


def create_diagnosis(
    db: Session, user_id: uuid.UUID, request_data: DiagnosisRequest
) -> Diagnosis:
    """
    Create a new diagnosis record with AI predictions.

    Args:
        db: Database session
        user_id: UUID of the authenticated user
        request_data: Diagnosis request with symptoms

    Returns:
        Created Diagnosis instance
    """
    # Generate predictions using mock AI
    predictions_data = mock_ai_diagnosis(request_data.symptoms)

    # Create diagnosis record
    diagnosis = Diagnosis(
        user_id=user_id,
        symptoms=request_data.symptoms,
        severity=request_data.severity,
        duration=request_data.duration,
        predictions=predictions_data,
    )

    db.add(diagnosis)
    db.commit()
    db.refresh(diagnosis)

    return diagnosis


def get_user_diagnosis_history(
    db: Session, user_id: uuid.UUID, page: int = 1, limit: int = 10
) -> tuple[List[Diagnosis], int]:
    """
    Get paginated diagnosis history for a user.

    Args:
        db: Database session
        user_id: UUID of the user
        page: Page number (1-indexed)
        limit: Items per page

    Returns:
        Tuple of (diagnosis list, total count)
    """
    # Ensure limit doesn't exceed 50
    limit = min(limit, 50)

    # Calculate offset
    offset = (page - 1) * limit

    # Query diagnoses
    query = db.query(Diagnosis).filter(Diagnosis.user_id == user_id)
    total = query.count()

    diagnoses = (
        query.order_by(Diagnosis.created_at.desc()).offset(offset).limit(limit).all()
    )

    return diagnoses, total


def get_diagnosis_by_id(
    db: Session, diagnosis_id: uuid.UUID, user_id: uuid.UUID
) -> Diagnosis:
    """
    Get a specific diagnosis by ID.

    Args:
        db: Database session
        diagnosis_id: UUID of the diagnosis
        user_id: UUID of the authenticated user

    Returns:
        Diagnosis instance

    Raises:
        HTTPException: 404 if not found or not owned by user
    """
    from fastapi import HTTPException, status

    diagnosis = (
        db.query(Diagnosis)
        .filter(Diagnosis.id == diagnosis_id, Diagnosis.user_id == user_id)
        .first()
    )

    if not diagnosis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Diagnosis not found"
        )

    return diagnosis
