"""
ML Service - Integration with trained disease diagnosis models
Loads Abhishek's ML models and provides prediction API
"""

from typing import List, Dict
import sys
from pathlib import Path

# Add ml_module to path
ml_module_path = Path(__file__).parent.parent / "ml_module"
sys.path.insert(0, str(ml_module_path))

from predict import DiseasePredictor


class MLDiagnosisService:
    """
    Service for making disease predictions using trained ML models
    """
    
    def __init__(self):
        """Initialize ML models on startup"""
        self.predictor = None
        self._load_models()
    
    def _load_models(self):
        """Load the trained Random Forest and XGBoost models"""
        try:
            print("🔄 Loading ML models...")
            self.predictor = DiseasePredictor()
            print("✅ ML models loaded successfully")
            print(f"   • Available symptoms: {len(self.predictor.get_available_symptoms())}")
            print(f"   • Available diseases: {len(self.predictor.get_available_diseases())}")
        except Exception as e:
            print(f"❌ Error loading ML models: {e}")
            self.predictor = None
    
    def is_available(self) -> bool:
        """Check if ML models are loaded and available"""
        return self.predictor is not None
    
    def predict_disease(self, symptoms: List[str]) -> List[Dict]:
        """
        Predict disease from symptoms using ML models
        
        Args:
            symptoms: List of symptom strings
            
        Returns:
            List of prediction dictionaries with disease, confidence, etc.
        """
        if not self.is_available():
            raise RuntimeError("ML models not loaded")
        
        # Validate symptoms
        validation = self.predictor.validate_symptoms(symptoms)
        
        if not validation['valid_symptoms']:
            return [{
                "disease": "Unable to diagnose",
                "confidence": 0.0,
                "severity": "unknown",
                "description": "No valid symptoms provided",
                "recommendations": [
                    "Please provide valid symptom names",
                    f"Available symptoms: {', '.join(self.predictor.get_available_symptoms()[:10])}...",
                ],
                "model_used": "validation",
                "valid_symptoms": [],
                "invalid_symptoms": validation['invalid_symptoms']
            }]
        
        # Get predictions from both models
        ensemble_result = self.predictor.predict_with_both_models(validation['valid_symptoms'])
        
        # Primary prediction (Random Forest)
        rf_pred = ensemble_result['random_forest']
        
        # Build response
        predictions = []
        
        # Primary prediction
        primary = {
            "disease": rf_pred['predicted_disease'],
            "confidence": rf_pred['confidence_percentage'] / 100,  # Convert to 0-1 scale
            "severity": self._determine_severity(rf_pred['confidence_percentage']),
            "description": self._get_disease_description(rf_pred['predicted_disease']),
            "recommendations": self._get_recommendations(rf_pred['predicted_disease']),
            "model_used": "random_forest",
            "valid_symptoms": validation['valid_symptoms'],
            "invalid_symptoms": validation['invalid_symptoms']
        }
        predictions.append(primary)
        
        # Add XGBoost prediction if different
        xgb_pred = ensemble_result['xgboost']
        if xgb_pred['predicted_disease'] != rf_pred['predicted_disease']:
            secondary = {
                "disease": xgb_pred['predicted_disease'],
                "confidence": xgb_pred['confidence_percentage'] / 100,
                "severity": self._determine_severity(xgb_pred['confidence_percentage']),
                "description": self._get_disease_description(xgb_pred['predicted_disease']),
                "recommendations": self._get_recommendations(xgb_pred['predicted_disease']),
                "model_used": "xgboost",
                "valid_symptoms": validation['valid_symptoms'],
                "invalid_symptoms": validation['invalid_symptoms']
            }
            predictions.append(secondary)
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return predictions
    
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


# Global ML service instance (loaded on startup)
ml_service = MLDiagnosisService()


def get_ml_diagnosis(symptoms: List[str]) -> List[Dict]:
    """
    Get ML-powered disease diagnosis from symptoms
    
    Args:
        symptoms: List of symptom strings
        
    Returns:
        List of prediction dictionaries
    """
    if not ml_service.is_available():
        # Fallback to mock diagnosis if ML not available
        from app.services.diagnosis_service import mock_ai_diagnosis
        print("⚠️  ML models not available, using mock diagnosis")
        return mock_ai_diagnosis(symptoms)
    
    try:
        return ml_service.predict_disease(symptoms)
    except Exception as e:
        print(f"❌ ML prediction error: {e}")
        # Fallback to mock
        from app.services.diagnosis_service import mock_ai_diagnosis
        return mock_ai_diagnosis(symptoms)


def is_ml_available() -> bool:
    """Check if ML models are loaded and ready"""
    return ml_service.is_available()
