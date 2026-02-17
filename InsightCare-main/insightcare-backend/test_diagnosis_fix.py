"""Test diagnosis creation with the fixed schema"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.database import SessionLocal
from app.models.diagnosis import Diagnosis
import uuid

# Create test diagnosis
db = SessionLocal()

test_diagnosis = Diagnosis(
    id=uuid.uuid4(),
    user_id=uuid.uuid4(),  # Random UUID for test
    symptoms=["fever", "cough", "headache"],  # Now as array of strings
    severity="moderate",
    duration="3 days",
    predictions=[
        {
            "disease": "Common Cold",
            "confidence": 0.85,
            "severity": "mild",
            "description": "Test",
            "recommendations": ["Rest", "Hydrate"],
        }
    ],
)

try:
    db.add(test_diagnosis)
    db.commit()
    print("✅ Successfully created diagnosis!")
    print(f"   ID: {test_diagnosis.id}")
    print(f"   Symptoms: {test_diagnosis.symptoms}")
    print(f"   Predictions: {len(test_diagnosis.predictions)} found")

    # Clean up test data
    db.delete(test_diagnosis)
    db.commit()
    print("✅ Test diagnosis removed")
except Exception as e:
    print(f"❌ Error: {e}")
    db.rollback()
finally:
    db.close()
