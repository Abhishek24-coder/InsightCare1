"""
ML Service - Hybrid Classical-Quantum Disease Diagnosis
Integrates Classical ML (RF + XGBoost) with Quantum ML (QSVM)
Uses weighted ensemble: 70% Classical + 30% Quantum
"""

from typing import List, Dict
import sys
from pathlib import Path

# Add ml_module to path
ml_module_path = Path(__file__).parent.parent / "ml_module"
sys.path.insert(0, str(ml_module_path))

from app.ml_module.hybrid_predictor import HybridPredictor


class MLDiagnosisService:
    """
    Service for hybrid classical-quantum disease predictions
    Uses weighted ensemble of RF + XGBoost + QSVM
    """
    
    def __init__(self, use_hybrid: bool = True):
        """
        Initialize ML models on startup
        
        Args:
            use_hybrid: Whether to use hybrid quantum-classical ensemble (default: True)
        """
        self.predictor = None
        self.use_hybrid = use_hybrid
        self._load_models()
    
    def _load_models(self):
        """Load the hybrid ensemble (Classical + Quantum models)"""
        try:
            print("🔄 Loading Hybrid Ensemble...")
            print("   Classical: Random Forest + XGBoost")
            print("   Quantum: QSVM (if available)")
            
            self.predictor = HybridPredictor()
            
            print("\n✅ Hybrid ensemble initialized")
            print(f"   • Available symptoms: {len(self.predictor.get_available_symptoms())}")
            print(f"   • Available diseases: {len(self.predictor.get_available_diseases())}")
            print(f"   • Quantum available: {self.predictor.is_quantum_available()}")
            
        except Exception as e:
            print(f"❌ Error loading hybrid ensemble: {e}")
            print("   Falling back to classical-only mode")
            self.predictor = None
    
    def is_available(self) -> bool:
        """Check if ML models are loaded and available"""
        return self.predictor is not None
    
    def is_quantum_available(self) -> bool:
        """Check if quantum model is available"""
        if not self.is_available():
            return False
        return self.predictor.is_quantum_available()
    
    def predict_disease(self, symptoms: List[str], use_quantum: bool = True) -> List[Dict]:
        """
        Predict disease using hybrid classical-quantum ensemble
        
        Args:
            symptoms: List of symptom strings
            use_quantum: Whether to use quantum model in ensemble (default: True)
            
        Returns:
            List of prediction dictionaries with disease, confidence, etc.
        """
        if not self.is_available():
            raise RuntimeError("ML models not loaded")
        
        # Use hybrid prediction
        hybrid_result = self.predictor.predict_hybrid(symptoms, use_quantum=use_quantum)
        
        # Check for errors
        if 'error' in hybrid_result:
            return [{
                "disease": hybrid_result['disease'],
                "confidence": hybrid_result['confidence'],
                "severity": "unknown",
                "description": hybrid_result['error'],
                "recommendations": [
                    "Please provide valid symptom names",
                    f"Available symptoms: {', '.join(self.predictor.get_available_symptoms()[:10])}...",
                ],
                "model_used": hybrid_result['ensemble_method'],
                "valid_symptoms": hybrid_result.get('valid_symptoms', []),
                "invalid_symptoms": hybrid_result.get('invalid_symptoms', []),
                "ensemble_info": {
                    "method": hybrid_result['ensemble_method'],
                    "quantum_used": False
                }
            }]
        
        # Build response with hybrid results
        disease = hybrid_result['disease']
        confidence = hybrid_result['confidence']
        
        prediction = {
            "disease": disease,
            "confidence": confidence,  # Already in 0-1 scale
            "severity": self._determine_severity(confidence * 100),
            "description": self._get_disease_description(disease),
            "recommendations": self._get_recommendations(disease),
            "model_used": hybrid_result['ensemble_method'],
            "valid_symptoms": hybrid_result.get('valid_symptoms', []),
            "invalid_symptoms": hybrid_result.get('invalid_symptoms', []),
            "ensemble_info": {
                "method": hybrid_result['ensemble_method'],
                "models_used": hybrid_result.get('models_used', []),
                "quantum_available": self.is_quantum_available(),
                "quantum_used": use_quantum and self.is_quantum_available(),
                "classical_prediction": hybrid_result.get('classical_prediction'),
                "quantum_prediction": hybrid_result.get('quantum_prediction'),
                "weights": hybrid_result.get('weights'),
                "voting_details": hybrid_result.get('voting_details')
            }
        }
        
        return [prediction]
    
    def _determine_severity(self, confidence: float) -> str:
        """Determine severity based on confidence level"""
        if confidence >= 70:
            return "high"
        elif confidence >= 40:
            return "moderate"
        else:
            return "low"
    
    def _get_disease_description(self, disease: str) -> str:
        """Get description for disease (from loaded data if available)"""
        try:
            description_dict = self.predictor.pipeline.description_dict
            return description_dict.get(disease, f"Medical condition: {disease}")
        except:
            return f"Medical condition: {disease}"
    
    def _get_recommendations(self, disease: str) -> List[str]:
        """Get recommendations for disease (from loaded data if available)"""
        try:
            precaution_dict = self.predictor.pipeline.precaution_dict
            precautions = precaution_dict.get(disease, [])
            
            if precautions:
                return [p for p in precautions if p and p.strip()]
            else:
                return [
                    "Consult a healthcare professional for proper diagnosis",
                    "Follow prescribed treatment plan",
                    "Monitor symptoms closely",
                    "Maintain good hygiene and rest"
                ]
        except:
            return [
                "Consult a healthcare professional",
                "Follow medical advice",
                "Rest and stay hydrated"
            ]
    
    def get_available_symptoms(self) -> List[str]:
        """Get list of all available symptoms"""
        if not self.is_available():
            return []
        return self.predictor.get_available_symptoms()
    
    def get_available_diseases(self) -> List[str]:
        """Get list of all available diseases"""
        if not self.is_available():
            return []
        return self.predictor.get_available_diseases()


# Global ML service instance with hybrid ensemble (loaded on startup)
ml_service = MLDiagnosisService(use_hybrid=True)


def get_ml_diagnosis(symptoms: List[str], use_quantum: bool = True) -> List[Dict]:
    """
    Get hybrid classical-quantum disease diagnosis from symptoms
    
    Args:
        symptoms: List of symptom strings
        use_quantum: Whether to use quantum model in ensemble (default: True)
        
    Returns:
        List of prediction dictionaries with ensemble results
    """
    if not ml_service.is_available():
        # Fallback to mock diagnosis if ML not available
        from app.services.diagnosis_service import mock_ai_diagnosis
        print("⚠️  Hybrid ensemble not available, using mock diagnosis")
        return mock_ai_diagnosis(symptoms)
    
    try:
        return ml_service.predict_disease(symptoms, use_quantum=use_quantum)
    except Exception as e:
        print(f"❌ Hybrid prediction error: {e}")
        # Fallback to mock
        from app.services.diagnosis_service import mock_ai_diagnosis
        return mock_ai_diagnosis(symptoms)


def is_ml_available() -> bool:
    """Check if ML models are loaded and ready"""
    return ml_service.is_available()


def is_quantum_available() -> bool:
    """Check if quantum model is available in the ensemble"""
    return ml_service.is_quantum_available()
