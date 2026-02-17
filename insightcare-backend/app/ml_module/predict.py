"""
Prediction API - Easy-to-use functions for disease prediction
Author: Abhishek
Description: High-level API for making predictions with trained models
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DiseasePredictor:
    """
    High-level API for disease prediction from symptoms
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the predictor with trained models
        
        Args:
            models_dir: Directory containing saved models
        """
        if models_dir is None:
            self.models_dir = Path(__file__).parent / "models"
        else:
            self.models_dir = Path(models_dir)
        
        # Models and encoders
        self.rf_model = None
        self.xgb_model = None
        self.label_encoder = None
        self.symptoms_list = None
        self.diseases_list = None
        self.severity_dict = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all trained models and encoders"""
        try:
            # Load Random Forest
            rf_path = self.models_dir / "random_forest_model.pkl"
            with open(rf_path, 'rb') as f:
                self.rf_model = pickle.load(f)
            
            # Load XGBoost
            xgb_path = self.models_dir / "xgboost_model.pkl"
            with open(xgb_path, 'rb') as f:
                self.xgb_model = pickle.load(f)
            
            # Load encoder data
            encoder_path = self.models_dir / "label_encoder.pkl"
            with open(encoder_path, 'rb') as f:
                encoder_data = pickle.load(f)
                self.label_encoder = encoder_data['label_encoder']
                self.symptoms_list = encoder_data['symptoms_list']
                self.diseases_list = encoder_data['diseases_list']
                self.severity_dict = encoder_data['severity_dict']
            
            print(f"✓ Models loaded successfully")
            print(f"  • Random Forest: {self.rf_model.n_estimators} trees")
            print(f"  • XGBoost: {self.xgb_model.n_estimators} estimators")
            print(f"  • Symptoms: {len(self.symptoms_list)}")
            print(f"  • Diseases: {len(self.diseases_list)}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Models not found. Please train models first. Error: {e}"
            )
        except Exception as e:
            raise Exception(f"Error loading models: {e}")
    
    def _create_feature_vector(self, symptoms: List[str]) -> np.ndarray:
        """
        Create feature vector from symptom list
        
        Args:
            symptoms: List of symptom names
            
        Returns:
            Feature vector array
        """
        # Initialize feature vector
        feature_vector = np.zeros(len(self.symptoms_list))
        
        # Clean and process symptoms
        for symptom in symptoms:
            symptom = str(symptom).strip().lower().replace(' ', '_')
            
            if symptom in self.symptoms_list:
                symptom_idx = self.symptoms_list.index(symptom)
                
                # Use severity weight if available
                if symptom in self.severity_dict:
                    feature_vector[symptom_idx] = self.severity_dict[symptom]
                else:
                    feature_vector[symptom_idx] = 1
        
        return feature_vector
    
    def predict(self, 
                symptoms: List[str], 
                model: str = 'random_forest',
                top_k: int = 5) -> Dict:
        """
        Predict disease from symptoms
        
        Args:
            symptoms: List of symptom names
            model: Which model to use ('random_forest' or 'xgboost')
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Validate input
        if not symptoms or len(symptoms) == 0:
            raise ValueError("Symptoms list cannot be empty")
        
        # Create feature vector
        feature_vector = self._create_feature_vector(symptoms)
        feature_vector = feature_vector.reshape(1, -1)
        
        # Select model
        if model.lower() == 'random_forest':
            selected_model = self.rf_model
        elif model.lower() == 'xgboost':
            selected_model = self.xgb_model
        else:
            raise ValueError(f"Invalid model: {model}. Use 'random_forest' or 'xgboost'")
        
        # Get predictions
        prediction = selected_model.predict(feature_vector)[0]
        probabilities = selected_model.predict_proba(feature_vector)[0]
        
        # Get top-k predictions
        top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        predictions = []
        for idx in top_k_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(probabilities[idx])
            predictions.append({
                'disease': disease,
                'confidence': confidence,
                'confidence_percentage': round(confidence * 100, 2)
            })
        
        # Main prediction
        main_disease = self.label_encoder.inverse_transform([prediction])[0]
        main_confidence = float(probabilities[prediction])
        
        return {
            'predicted_disease': main_disease,
            'confidence': main_confidence,
            'confidence_percentage': round(main_confidence * 100, 2),
            'model_used': model,
            'input_symptoms': symptoms,
            'top_predictions': predictions
        }
    
    def predict_with_both_models(self, symptoms: List[str], top_k: int = 3) -> Dict:
        """
        Get predictions from both Random Forest and XGBoost
        
        Args:
            symptoms: List of symptom names
            top_k: Number of top predictions per model
            
        Returns:
            Dictionary with predictions from both models
        """
        rf_result = self.predict(symptoms, model='random_forest', top_k=top_k)
        xgb_result = self.predict(symptoms, model='xgboost', top_k=top_k)
        
        # Check if both models agree
        agreement = rf_result['predicted_disease'] == xgb_result['predicted_disease']
        
        return {
            'random_forest': rf_result,
            'xgboost': xgb_result,
            'models_agree': agreement,
            'consensus_disease': rf_result['predicted_disease'] if agreement else None,
            'input_symptoms': symptoms
        }
    
    def get_disease_info(self, disease: str) -> Dict:
        """
        Get information about a disease
        
        Args:
            disease: Disease name
            
        Returns:
            Dictionary with disease information
        """
        from data_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        pipeline.load_data()
        info = pipeline.get_disease_info(disease)
        
        return info
    
    def get_available_symptoms(self) -> List[str]:
        """
        Get list of all available symptoms
        
        Returns:
            List of symptom names
        """
        return self.symptoms_list.copy()
    
    def get_available_diseases(self) -> List[str]:
        """
        Get list of all available diseases
        
        Returns:
            List of disease names
        """
        return self.diseases_list.copy()
    
    def validate_symptoms(self, symptoms: List[str]) -> Dict:
        """
        Validate if symptoms are in the training data
        
        Args:
            symptoms: List of symptom names
            
        Returns:
            Dictionary with validation results
        """
        valid_symptoms = []
        invalid_symptoms = []
        
        for symptom in symptoms:
            symptom_clean = str(symptom).strip().lower().replace(' ', '_')
            if symptom_clean in self.symptoms_list:
                valid_symptoms.append(symptom_clean)
            else:
                invalid_symptoms.append(symptom)
        
        return {
            'valid_symptoms': valid_symptoms,
            'invalid_symptoms': invalid_symptoms,
            'all_valid': len(invalid_symptoms) == 0,
            'validation_message': f"{len(valid_symptoms)}/{len(symptoms)} symptoms are valid"
        }


def predict_disease(symptoms: List[str], model: str = 'random_forest') -> Dict:
    """
    Standalone function for quick disease prediction
    
    Args:
        symptoms: List of symptom names
        model: Which model to use ('random_forest' or 'xgboost')
        
    Returns:
        Prediction results
    """
    predictor = DiseasePredictor()
    return predictor.predict(symptoms, model=model)


def main():
    """Test the prediction API"""
    print("="*70)
    print("TESTING PREDICTION API")
    print("="*70)
    
    # Initialize predictor
    predictor = DiseasePredictor()
    
    # Test cases
    test_cases = [
        {
            'name': 'Diabetes',
            'symptoms': ['fatigue', 'weight_loss', 'increased_appetite', 'polyuria', 'excessive_hunger']
        },
        {
            'name': 'Common Cold',
            'symptoms': ['continuous_sneezing', 'chills', 'runny_nose', 'cough']
        },
        {
            'name': 'Malaria',
            'symptoms': ['chills', 'vomiting', 'high_fever', 'sweating', 'headache', 'nausea', 'muscle_pain']
        },
        {
            'name': 'Fungal Infection',
            'symptoms': ['itching', 'skin_rash', 'nodal_skin_eruptions']
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: {case['name']}")
        print(f"{'='*70}")
        print(f"\n📋 Input Symptoms: {', '.join(case['symptoms'])}")
        
        # Validate symptoms
        validation = predictor.validate_symptoms(case['symptoms'])
        print(f"\n✓ Validation: {validation['validation_message']}")
        
        # Get prediction with Random Forest
        print(f"\n🌲 Random Forest Prediction:")
        rf_result = predictor.predict(case['symptoms'], model='random_forest', top_k=3)
        print(f"   Primary: {rf_result['predicted_disease']} ({rf_result['confidence_percentage']}%)")
        print(f"   Top 3 Predictions:")
        for pred in rf_result['top_predictions']:
            print(f"     • {pred['disease']:40s} {pred['confidence_percentage']:5.2f}%")
        
        # Get prediction with XGBoost
        print(f"\n🚀 XGBoost Prediction:")
        xgb_result = predictor.predict(case['symptoms'], model='xgboost', top_k=3)
        print(f"   Primary: {xgb_result['predicted_disease']} ({xgb_result['confidence_percentage']}%)")
        print(f"   Top 3 Predictions:")
        for pred in xgb_result['top_predictions']:
            print(f"     • {pred['disease']:40s} {pred['confidence_percentage']:5.2f}%")
        
        # Check agreement
        if rf_result['predicted_disease'] == xgb_result['predicted_disease']:
            print(f"\n✅ Both models agree: {rf_result['predicted_disease']}")
        else:
            print(f"\n⚠️  Models disagree:")
            print(f"   RF: {rf_result['predicted_disease']}")
            print(f"   XGB: {xgb_result['predicted_disease']}")
    
    # Test invalid symptoms
    print(f"\n{'='*70}")
    print("TESTING INVALID SYMPTOMS")
    print(f"{'='*70}")
    
    invalid_test = ['fever', 'invalid_symptom', 'cough', 'fake_symptom']
    validation = predictor.validate_symptoms(invalid_test)
    print(f"\nTest symptoms: {invalid_test}")
    print(f"Valid: {validation['valid_symptoms']}")
    print(f"Invalid: {validation['invalid_symptoms']}")
    print(f"Status: {'✅ All valid' if validation['all_valid'] else '⚠️  Some invalid'}")
    
    print(f"\n{'='*70}")
    print("✅ PREDICTION API TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
