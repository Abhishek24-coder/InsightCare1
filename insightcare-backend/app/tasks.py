# app/tasks.py
import os
from celery import Celery
from app.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND
from app.database import SessionLocal
from app.models.diagnosis_result import DiagnosisResult
from app.quantum_module.quantum_ai import run_quantum_simulation

celery = Celery("app.tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


@celery.task(bind=True)
def run_diagnosis_task(self, diagnosis_id: int):
    db = SessionLocal()
    try:
        diag = db.query(DiagnosisResult).get(diagnosis_id)
        if not diag:
            return {"error": "Diagnosis not found"}
        diag.status = "running"
        db.add(diag)
        db.commit()
        payload = diag.input_payload or {}
        result = run_quantum_simulation(payload)
        diag.output_payload = result
        diag.status = "completed"
        db.add(diag)
        db.commit()
        return {"status": "completed", "result": result}
    except Exception as e:
        try:
            diag.status = "failed"
            diag.output_payload = {"error": str(e)}
            db.add(diag)
            db.commit()
        except Exception:
            pass
        return {"status": "failed", "error": str(e)}
    finally:
        db.close()
