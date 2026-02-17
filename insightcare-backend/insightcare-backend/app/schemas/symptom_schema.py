# app/schemas/symptom_schema.py
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict
import bleach


def sanitize_text(s: str) -> str:
    return bleach.clean(s or "", tags=[], strip=True)


class SymptomCreate(BaseModel):
    patient_id: int
    text: str = Field(..., min_length=1, max_length=2000)
    structured: Optional[Dict] = {}
    severity: Optional[str] = Field(None, regex="^(mild|moderate|severe|unknown)$")

    @validator("text")
    def clean_text(cls, v):
        return sanitize_text(v)


class SymptomResponse(SymptomCreate):
    id: int
    reported_by: Optional[int]
    created_at: Optional[str]

    class Config:
        orm_mode = True
