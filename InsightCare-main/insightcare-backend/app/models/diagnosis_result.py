# app/models/diagnosis_result.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, func
from app.database import Base


class DiagnosisResult(Base):
    __tablename__ = "diagnosis_results"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    run_by = Column(Integer, nullable=True)
    input_payload = Column(JSON, default={})
    output_payload = Column(JSON, default={})
    model = Column(String(100), default="quantum_v1")
    status = Column(
        String(50), default="pending"
    )  # pending, running, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
