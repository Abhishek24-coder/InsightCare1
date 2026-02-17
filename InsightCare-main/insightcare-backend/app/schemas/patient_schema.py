# app/schemas/patient_schema.py
from pydantic import BaseModel
from typing import Optional


class PatientCreate(BaseModel):
    name: str
    age: Optional[int]
    blood_pressure: Optional[float] = None
    cholesterol: Optional[float] = None
    diagnosis: Optional[str] = None


class PatientResponse(PatientCreate):
    id: int

    class Config:
        orm_mode = True
