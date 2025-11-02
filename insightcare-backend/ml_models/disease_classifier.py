"""
Production-Ready Disease Classifier
Use this class in FastAPI for predictions
"""

import joblib
import os
import numpy as np
from typing import List, Dict, Any

class DiseaseClassifier:
    """
    Disease prediction from symptoms
    
    Usage:
        classifier = DiseaseClassifier()
        result = classifier.predict(['fever', 'cough', 'headache'])
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize classifier
        
        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = model_dir
        self.model = None
        self.encoder = None
        self.label_encoder = None
        self.metadata = None
        self.is_loaded = False
        
    def load_model(self, model_name: str = 'best_model.pkl'):
        """
        Load trained model from disk
        
        Args:
            model_name: Name of model file (default: best_model.pkl)
        """
        try:
            # Load model
            model_path = os.path.join(self.model_dir, model_name)
            self.model = joblib.load(model_path)
            
            # Load symptom encoder
            encoder_path = os.path.join(self.model_dir, 'symptom_encoder.pkl')
            self.encoder = joblib.load(encoder_path)
            
            # Load label encoder (for XGBoost)
            le_path = os.path.join(self.model_dir, 'label_encoder.pkl')
            if os.path.exists(le_path):
                self.label_encoder = joblib.load(le_path)
            
            # Load metadata
            meta_path = os.path.join(self.model_dir, 'metadata.pkl')
            if os.path.exists(meta_path):
                self.metadata = joblib.load(meta_path)
            
            self.is_loaded = True
            print(f"✅ Model loaded: {model_name}")
            if self.metadata:
                print(f"   Best model: {self.metadata.get('best_model', 'Unknown')}")
                print(f"   Accuracy: {self.metadata.get('best_accuracy', 0)*100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, symptoms: List[str]) -> Dict[str, Any]:
        """
        Predict disease from symptoms
        
        Args:
            symptoms: List of symptom strings, e.g., ['fever', 'cough', 'headache']
        
        Returns:
            {
                'disease': 'Malaria',
                'confidence': 0.87,
                'top_3': [
                    {'disease': 'Malaria', 'probability': 0.87},
                    {'disease': 'Dengue', 'probability': 0.09},
                    {'disease': 'Typhoid', 'probability': 0.04}
                ],
                'all_symptoms_used': ['fever', 'cough', 'headache'],
                'recommendation': 'Consult a doctor for confirmation'
            }
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not symptoms or len(symptoms) == 0:
            raise ValueError("No symptoms provided")
        
        # Clean symptoms (lowercase, strip spaces)
        symptoms_clean = [s.lower().strip().replace(' ', '_') for s in symptoms]
        
        # Encode symptoms
        symptoms_encoded = self.encoder.transform([symptoms_clean])
        
        # Predict
        if self.label_encoder:  # XGBoost model
            pred_encoded = self.model.predict(symptoms_encoded)[0]
            prediction = self.label_encoder.inverse_transform([pred_encoded])[0]
            probabilities = self.model.predict_proba(symptoms_encoded)[0]
            
            # Get top 3
            top_3_idx = probabilities.argsort()[-3:][::-1]
            top_3 = [
                {
                    'disease': self.label_encoder.inverse_transform([i])[0],
                    'probability': float(probabilities[i])
                }
                for i in top_3_idx
            ]
        else:  # Random Forest model
            prediction = self.model.predict(symptoms_encoded)[0]
            probabilities = self.model.predict_proba(symptoms_encoded)[0]
            
            # Get top 3
            top_3_idx = probabilities.argsort()[-3:][::-1]
            top_3 = [
                {
                    'disease': self.model.classes_[i],
                    'probability': float(probabilities[i])
                }
                for i in top_3_idx
            ]
        
        confidence = float(max(probabilities))
        
        # Generate recommendation
        if confidence >= 0.8:
            confidence_level = "high confidence"
        elif confidence >= 0.6:
            confidence_level = "moderate confidence"
        else:
            confidence_level = "low confidence"
        
        recommendation = (
            f"Based on your symptoms with {confidence_level}, you may have {prediction}. "
            f"This is an AI prediction and should not replace professional medical advice. "
            f"Please consult a healthcare provider for proper diagnosis and treatment."
        )
        
        return {
            'disease': prediction,
            'confidence': confidence,
            'top_3': top_3,
            'all_symptoms_used': symptoms_clean,
            'recommendation': recommendation
        }
    
    def predict_batch(self, symptoms_list: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Predict diseases for multiple patients
        
        Args:
            symptoms_list: List of symptom lists
        
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(symptoms) for symptoms in symptoms_list]
    
    def get_available_symptoms(self) -> List[str]:
        """Get list of all symptoms the model knows"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return sorted(self.encoder.classes_)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        if not self.metadata:
            return {'status': 'No metadata available'}
        
        return {
            'best_model': self.metadata.get('best_model', 'Unknown'),
            'accuracy': f"{self.metadata.get('best_accuracy', 0)*100:.2f}%",
            'rf_accuracy': f"{self.metadata.get('rf_accuracy', 0)*100:.2f}%",
            'xgb_accuracy': f"{self.metadata.get('xgb_accuracy', 0)*100:.2f}%",
            'n_diseases': self.metadata.get('n_diseases', 0),
            'n_symptoms': self.metadata.get('n_symptoms', 0),
            'n_samples': self.metadata.get('n_samples', 0),
            'training_date': self.metadata.get('training_date', 'Unknown')
        }


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    print("=" * 80)
    print("🧪 TESTING DISEASE CLASSIFIER")
    print("=" * 80)
    
    # Initialize and load model
    classifier = DiseaseClassifier()
    classifier.load_model()
    
    # Get model info
    print("\n📊 Model Information:")
    info = classifier.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test predictions
    print("\n" + "=" * 80)
    print("🔮 TEST PREDICTIONS")
    print("=" * 80)
    
    test_cases = [
        {
            'name': 'Case 1: Skin condition',
            'symptoms': ['itching', 'skin_rash', 'nodal_skin_eruptions']
        },
        {
            'name': 'Case 2: Respiratory symptoms',
            'symptoms': ['continuous_sneezing', 'chills', 'fatigue']
        },
        {
            'name': 'Case 3: Digestive issues',
            'symptoms': ['stomach_pain', 'acidity', 'vomiting', 'indigestion']
        },
        {
            'name': 'Case 4: Fever symptoms',
            'symptoms': ['high_fever', 'headache', 'nausea', 'muscle_pain']
        }
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print(f"Symptoms: {test['symptoms']}")
        
        result = classifier.predict(test['symptoms'])
        
        print(f"\n🎯 Prediction: {result['disease']}")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
        print(f"\n   Top 3 Possibilities:")
        for i, item in enumerate(result['top_3'], 1):
            print(f"   {i}. {item['disease']:<30} {item['probability']*100:>5.1f}%")
        print(f"\n   💡 Recommendation:")
        print(f"   {result['recommendation']}")
        print("-" * 80)
    
    # Show available symptoms (first 20)
    print(f"\n📋 Available Symptoms (showing first 20):")
    symptoms = classifier.get_available_symptoms()
    for i, symptom in enumerate(symptoms[:20], 1):
        print(f"   {i:2d}. {symptom}")
    print(f"   ... and {len(symptoms) - 20} more symptoms")
    
    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE!")
    print("=" * 80)
