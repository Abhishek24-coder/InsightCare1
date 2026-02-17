# app/routes/quantum_routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.patient import Patient
from app.quantum_module.quantum_ai import run_quantum_simulation
from app.utils.auth_helpers import get_current_user

router = APIRouter(prefix="/api/quantum", tags=["Quantum AI"])


@router.get("/simulate/{patient_id}")
def simulate(
    patient_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)
):
    patient = db.query(Patient).get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    patient_data = {
        "age": patient.age,
        "blood_pressure": patient.blood_pressure,
        "cholesterol": patient.cholesterol,
    }
    result = run_quantum_simulation(patient_data)
    return {"patient_id": patient.id, "simulation_result": result}
