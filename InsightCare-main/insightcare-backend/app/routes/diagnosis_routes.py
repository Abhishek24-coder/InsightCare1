# app/routes/diagnosis_routes.py
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.diagnosis_schema import DiagnosisRequest, DiagnosisResponse
from app.models.diagnosis_result import DiagnosisResult
from app.models.patient import Patient
from app.models.audit import AuditLog
from app.utils.auth_helpers import get_current_user
from app.tasks import run_diagnosis_task
from typing import Optional

router = APIRouter(prefix="/api/diagnosis", tags=["Diagnosis"])


@router.post("/", response_model=DiagnosisResponse, status_code=201)
def request_diagnosis(
    payload: DiagnosisRequest,
    request: Request,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    if not payload.patient_id and not payload.symptoms:
        raise HTTPException(status_code=400, detail="Provide patient_id or symptoms")

    # build input payload
    if payload.patient_id:
        patient = db.query(Patient).get(payload.patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        input_payload = {
            "patient_id": patient.id,
            "age": patient.age,
            "blood_pressure": patient.blood_pressure,
            "cholesterol": patient.cholesterol,
        }
    else:
        input_payload = payload.symptoms

    diag = DiagnosisResult(
        patient_id=payload.patient_id or -1,
        run_by=user.id,
        input_payload=input_payload,
        model=payload.prefer_model,
        status="pending",
    )
    db.add(diag)
    db.commit()
    db.refresh(diag)

    if payload.run_async:
        celery_task = run_diagnosis_task.delay(diag.id)
        audit = AuditLog(
            user_id=user.id,
            action="start_diagnosis_task",
            resource=f"diagnosis:{diag.id}",
            ip=request.client.host,
            metadata={"celery_task_id": celery_task.id},
        )
        db.add(audit)
        db.commit()
        return DiagnosisResponse(
            diagnosis_id=diag.id,
            patient_id=diag.patient_id,
            model=diag.model,
            status="scheduled",
            output={},
        )
    else:
        res = run_diagnosis_task.apply(args=[diag.id]).get()
        db.refresh(diag)
        return DiagnosisResponse(
            diagnosis_id=diag.id,
            patient_id=diag.patient_id,
            model=diag.model,
            status=diag.status,
            output=diag.output_payload or {},
        )
