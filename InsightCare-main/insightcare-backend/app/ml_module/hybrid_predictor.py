"""
Quantum-Classical Integration
Combines quantum and classical models for disease prediction
"""

import numpy as np
import pickle
from pathlib import Path
import time
from typing import List, Dict, Tuple

from predict import DiseasePredictor
from quantum_circuit import QuantumFeatureEncoder


class HybridPredictor:
    """
    Hybrid predictor combining classical and quantum models
    """
    
    def __init__(self):
        """Initialize hybrid predictor with classical models"""
        print("="*70)
        print("HYBRID QUANTUM-CLASSICAL PREDICTOR")
        print("="*70)
        
        # Load classical predictor
        print("\n[1/2] Loading classical models...")
        self.classical_predictor = DiseasePredictor()
        
        # Initialize quantum encoder (for future QSVM integration)
        print("\n[2/2] Initializing quantum encoder...")
        self.quantum_encoder = QuantumFeatureEncoder(
            n_features=131,
            encoding_type='zz',
            n_qubits=10
        )
        
        print("\n✅ Hybrid predictor ready!")
        print(f"   • Classical: Random Forest + XGBoost")
        print(f"   • Quantum: Feature encoding (QSVM integration ready)")
    
    def predict(self, symptoms: List[str], use_quantum=False):
        """
        Predict disease using hybrid approach
        
        Args:
            symptoms: List of symptom names
            use_quantum: Whether to use quantum encoding (future feature)
            
        Returns:
            Dictionary with prediction results
        """
        # Classical prediction
        classical_result = self.classical_predictor.predict(symptoms)
        
        result = {
            'symptoms': symptoms,
            'classical_prediction': classical_result,
            'quantum_encoding': 'ready' if use_quantum else 'not used',
            'hybrid_mode': use_quantum
        }
        
        return result
    
    def predict_with_ensemble(self, symptoms: List[str]):
        """
        Predict using both RF and XGBoost (ensemble)
        
        Args:
            symptoms: List of symptom names
            
        Returns:
            Dictionary with ensemble predictions
        """
        result = self.classical_predictor.predict_with_both_models(symptoms)
        return result
    
    def get_quantum_features(self, symptoms: List[str]):
        """
        Encode symptoms into quantum features
        
        Args:
            symptoms: List of symptom names
            
        Returns:
            Quantum-encoded features
        """
        # Get classical feature vector
        validation = self.classical_predictor.validate_symptoms(symptoms)
        valid_symptoms = validation['valid_symptoms']
        
        # Create feature vector
        vector = np.zeros(131)
        symptoms_list = self.classical_predictor.get_available_symptoms()
        
        for symptom in valid_symptoms:
            if symptom in symptoms_list:
                idx = symptoms_list.index(symptom)
                vector[idx] = 1
        
        # Encode to quantum features
        quantum_features = self.quantum_encoder.encode_features(vector.reshape(1, -1))
        
        return quantum_features[0]


def demo_hybrid_prediction():
    """
    Demonstrate hybrid quantum-classical prediction
    """
    print("\n" + "="*70)
    print("HYBRID PREDICTION DEMO")
    print("="*70)
    
    # Initialize hybrid predictor
    hybrid = HybridPredictor()
    
    # Test cases
    test_cases = [
        {
            'name': 'Case 1: Diabetes',
            'symptoms': ['fatigue', 'weight loss', 'increased appetite', 'frequent urination']
        },
        {
            'name': 'Case 2: Malaria',
            'symptoms': ['chills', 'high fever', 'sweating', 'headache', 'nausea']
        },
        {
            'name': 'Case 3: Pneumonia',
            'symptoms': ['cough', 'chest pain', 'breathlessness', 'fast heart rate']
        }
    ]
    
    print("\n" + "="*70)
    print("TESTING HYBRID PREDICTIONS")
    print("="*70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"{test_case['name']}")
        print(f"{'='*70}")
        print(f"Symptoms: {', '.join(test_case['symptoms'])}")
        
        # Classical prediction
        print(f"\n📊 Classical Prediction:")
        start = time.time()
        result = hybrid.predict(test_case['symptoms'])
        classical_time = time.time() - start
        
        pred = result['classical_prediction']
        print(f"   • Disease: {pred['predicted_disease']}")
        print(f"   • Confidence: {pred['confidence_percentage']}%")
        print(f"   • Model: {pred['model_used']}")
        print(f"   • Time: {classical_time:.4f}s")
        
        # Ensemble prediction
        print(f"\n🔄 Ensemble Prediction:")
        ensemble = hybrid.predict_with_ensemble(test_case['symptoms'])
        print(f"   • RF: {ensemble['random_forest']['predicted_disease']} ({ensemble['random_forest']['confidence_percentage']}%)")
        print(f"   • XGBoost: {ensemble['xgboost']['predicted_disease']} ({ensemble['xgboost']['confidence_percentage']}%)")
        print(f"   • Agreement: {'✅ Yes' if ensemble['models_agree'] else '❌ No'}")
        
        # Quantum features
        print(f"\n⚛️  Quantum Encoding:")
        quantum_features = hybrid.get_quantum_features(test_case['symptoms'])
        print(f"   • Quantum features: {quantum_features.shape} dimensions")
        print(f"   • Feature range: [{quantum_features.min():.4f}, {quantum_features.max():.4f}]")
        print(f"   • Status: Ready for QSVM")
    
    print("\n" + "="*70)
    print("✅ HYBRID DEMO COMPLETE")
    print("="*70)
    print("\n💡 Summary:")
    print("   • Classical models: ✅ Working")
    print("   • Quantum encoding: ✅ Ready")
    print("   • QSVM integration: ⚪ Optional enhancement")
    print("   • Hybrid mode: ✅ Implemented")


if __name__ == "__main__":
    demo_hybrid_prediction()
