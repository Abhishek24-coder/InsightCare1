# app/routes/symptom_routes.py
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.schemas.symptom_schema import SymptomCreate, SymptomResponse
from app.models.symptom import Symptom
from app.models.patient import Patient
from app.models.audit import AuditLog
from app.utils.auth_helpers import get_current_user

router = APIRouter(prefix="/api/symptoms", tags=["Symptoms"])


@router.post("/", response_model=SymptomResponse, status_code=201)
def create_symptom(
    payload: SymptomCreate,
    request: Request,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    patient = db.query(Patient).get(payload.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    symptom = Symptom(
        patient_id=payload.patient_id,
        reported_by=user.id,
        text=payload.text,
        structured=payload.structured or {},
        severity=payload.severity,
    )
    db.add(symptom)
    db.commit()
    db.refresh(symptom)

    audit = AuditLog(
        user_id=user.id,
        action="create_symptom",
        resource=f"patient:{patient.id}",
        ip=request.client.host,
        metadata={"symptom_id": symptom.id},
    )
    db.add(audit)
    db.commit()

    return symptom


@router.get("/patient/{patient_id}", response_model=List[SymptomResponse])
def get_symptoms(
    patient_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)
):
    items = (
        db.query(Symptom)
        .filter(Symptom.patient_id == patient_id)
        .order_by(Symptom.created_at.desc())
        .all()
    )
    return items
