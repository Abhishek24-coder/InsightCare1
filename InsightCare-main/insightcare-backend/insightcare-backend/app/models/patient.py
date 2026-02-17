# app/models/patient.py
from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from app.database import Base


class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    age = Column(Integer, nullable=True)
    blood_pressure = Column(Float, nullable=True)
    cholesterol = Column(Float, nullable=True)
    diagnosis = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime)
