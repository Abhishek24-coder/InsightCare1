# Simple ML API Server - No Database Required
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="InsightCare ML API",
    description="Machine Learning prediction service",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://insight-care-rust.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class DiagnosisRequest(BaseModel):
    symptoms: List[str]

class Prediction(BaseModel):
    disease: str
    confidence: float
    description: Optional[str] = None

class DiagnosisResponse(BaseModel):
    predictions: List[Prediction]
    model_used: str
    success: bool
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    ml_available: bool
    models_loaded: dict

# Global ML service
ml_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize ML models on startup"""
    global ml_service
    try:
        from app.services.ml_service import MLDiagnosisService
        ml_service = MLDiagnosisService()
        print("✅ ML models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load ML models: {e}")
        ml_service = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "InsightCare ML API",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/api/ml/health")
async def ml_health():
    """Check ML service health"""
    if ml_service is None:
        return HealthResponse(
            status="ML models not loaded",
            ml_available=False,
            models_loaded={}
        )
    
    try:
        health = ml_service.health_check()
        return HealthResponse(
            status="healthy",
            ml_available=health.get("ml_available", False),
            models_loaded=health.get("models_loaded", {})
        )
    except Exception as e:
        return HealthResponse(
            status=f"error: {str(e)}",
            ml_available=False,
            models_loaded={}
        )

@app.post("/api/ml/diagnose")
async def diagnose(request: DiagnosisRequest):
    """Predict disease from symptoms"""
    print(f"\n🔍 Received diagnosis request with {len(request.symptoms)} symptoms: {request.symptoms}")
    
    if ml_service is None:
        print("❌ ML service not available")
        raise HTTPException(
            status_code=503,
            detail="ML models not available"
        )
    
    if not request.symptoms or len(request.symptoms) == 0:
        print("❌ No symptoms provided")
        raise HTTPException(
            status_code=400,
            detail="No symptoms provided"
        )
    
    try:
        # Get predictions from ML service
        predictions_list = ml_service.predict_disease(
            symptoms=request.symptoms,
            use_quantum=True  # Use hybrid ensemble
        )
        
        # Transform to match frontend format
        predictions = []
        for pred in predictions_list:
            predictions.append({
                "disease": pred["disease"],
                "confidence": pred["confidence"] * 100,  # Convert to percentage
                "severity": pred.get("severity", "moderate"),
                "description": pred.get("description", ""),
                "recommendations": pred.get("recommendations", []),
                "model_used": pred.get("model_used", "hybrid_ensemble"),
                "valid_symptoms": pred.get("valid_symptoms", []),
                "invalid_symptoms": pred.get("invalid_symptoms", [])
            })
        
        return {
            "success": True,
            "predictions": predictions,
            "total_predictions": len(predictions),
            "ml_available": True,
            "message": "Diagnosis completed successfully"
        }
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "ml_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
