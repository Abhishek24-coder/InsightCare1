"""
Simple ML API Server for Testing
Run with: python -m uvicorn test_ml_server:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add parent directory to path for imports
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Import ML service
from app.services.ml_service import ml_service, get_ml_diagnosis
from app.api.ml_api import router as ml_router

# Create FastAPI app
app = FastAPI(
    title="InsightCare ML API (Test Server)",
    description="Machine Learning disease prediction API - Standalone test version",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include ML router
app.include_router(ml_router, prefix="/api")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "InsightCare ML API Test Server",
        "status": "running",
        "ml_loaded": ml_service.is_available(),
        "endpoints": {
            "predict": "/api/ml/diagnose",
            "health": "/api/ml/health",
            "symptoms": "/api/ml/symptoms",
            "diseases": "/api/ml/diseases",
            "docs": "/docs"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Print startup info"""
    print("=" * 60)
    print("🚀 InsightCare ML API Test Server Starting...")
    print("=" * 60)
    if ml_service.is_available():
        symptoms = ml_service.get_available_symptoms()
        diseases = ml_service.get_available_diseases()
        print(f"✅ ML Models Loaded:")
        print(f"   - Random Forest (100 trees)")
        print(f"   - XGBoost (100 estimators)")
        print(f"📊 Available Symptoms: {len(symptoms)}")
        print(f"🏥 Predictable Diseases: {len(diseases)}")
    else:
        print("❌ ML Models Failed to Load")
    print("=" * 60)
    print("📡 API Endpoints:")
    print("   POST /api/ml/diagnose - Get disease prediction")
    print("   GET  /api/ml/health   - Check system status")
    print("   GET  /api/ml/symptoms - List all symptoms")
    print("   GET  /api/ml/diseases - List all diseases")
    print("   GET  /docs            - API documentation")
    print("=" * 60)
    print("✨ Server ready! Visit http://localhost:8000/docs")
    print("=" * 60)
