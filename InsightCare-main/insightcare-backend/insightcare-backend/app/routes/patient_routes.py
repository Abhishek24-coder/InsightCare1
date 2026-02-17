# app/routes/patient_routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.patient import Patient
from app.schemas.patient_schema import PatientCreate, PatientResponse
from app.utils.auth_helpers import get_current_user, require_roles

router = APIRouter(prefix="/api/patient", tags=["Patients"])


@router.get("/", response_model=List[PatientResponse])
def list_patients(db: Session = Depends(get_db), user=Depends(get_current_user)):
    # Any authenticated user can list (adjust as needed)
    return db.query(Patient).all()


@router.post("/", response_model=PatientResponse, status_code=201)
def create_patient(
    payload: PatientCreate,
    db: Session = Depends(get_db),
    user=Depends(require_roles("doctor", "admin")),
):
    new = Patient(**payload.dict())
    db.add(new)
    db.commit()
    db.refresh(new)
    return new


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(
    patient_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)
):
    p = db.query(Patient).get(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return p
