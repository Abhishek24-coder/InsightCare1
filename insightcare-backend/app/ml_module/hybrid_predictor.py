"""
Quantum-Classical Hybrid Ensemble for Disease Prediction
Combines Classical ML (Random Forest + XGBoost) with Quantum ML (QSVM)
Uses weighted voting: 70% Classical + 30% Quantum
"""

import numpy as np
import pickle
from pathlib import Path
import time
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from predict import DiseasePredictor
from quantum_circuit import QuantumFeatureEncoder


class HybridPredictor:
    """
    Hybrid Ensemble: Classical ML (RF + XGBoost) + Quantum ML (QSVM)
    
    Voting Strategy:
    - Classical models (RF + XGBoost): 70% weight
    - Quantum model (QSVM): 30% weight
    - Weighted ensemble for final prediction
    """
    
    def __init__(self, models_dir: str = None):
        """Initialize hybrid predictor with classical and quantum models"""
        print("="*70)
        print("HYBRID QUANTUM-CLASSICAL ENSEMBLE PREDICTOR")
        print("="*70)
        
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        else:
            models_dir = Path(models_dir)
        
        self.models_dir = models_dir
        
        # Load classical predictor
        print("\n[1/3] Loading classical models...")
        self.classical_predictor = DiseasePredictor(models_dir=str(models_dir))
        print("✅ Random Forest + XGBoost loaded")
        
        # Initialize quantum encoder
        print("\n[2/3] Initializing quantum encoder...")
        self.quantum_encoder = QuantumFeatureEncoder(
            n_features=131,
            encoding_type='zz',
            n_qubits=10
        )
        print("✅ Quantum feature encoder ready")
        
        # Try to load QSVM model
        print("\n[3/3] Loading quantum QSVM model...")
        self.quantum_predictor = None
        self.qsvm_available = False
        
        try:
            qsvm_path = models_dir / "qsvm_model.pkl"
            if qsvm_path.exists():
                from qsvm_model import QSVMClassifier
                self.quantum_predictor = QSVMClassifier.load_model(qsvm_path)
                self.qsvm_available = True
                print("✅ QSVM model loaded successfully")
            else:
                print(f"⚠️  QSVM model not found at: {qsvm_path}")
                print("   Hybrid mode will use classical models only")
        except Exception as e:
            print(f"⚠️  Could not load QSVM: {e}")
            print("   Falling back to classical-only mode")
        
        # Ensemble weights
        self.classical_weight = 0.7
        self.quantum_weight = 0.3
        
        print("\n" + "="*70)
        print("✅ HYBRID PREDICTOR INITIALIZED")
        print("="*70)
        print(f"📊 Models Loaded:")
        print(f"   • Random Forest: ✅")
        print(f"   • XGBoost: ✅")
        print(f"   • QSVM: {'✅' if self.qsvm_available else '❌'}")
        print(f"\n⚖️  Ensemble Weights:")
        print(f"   • Classical (RF+XGBoost): {self.classical_weight*100:.0f}%")
        print(f"   • Quantum (QSVM): {self.quantum_weight*100:.0f}%")
        print("="*70)
    
    def predict_hybrid(self, symptoms: List[str], use_quantum: bool = True) -> Dict:
        """
        Hybrid ensemble prediction combining classical and quantum models
        
        Args:
            symptoms: List of symptom strings
            use_quantum: Whether to use quantum model (if available)
            
        Returns:
            Dictionary with hybrid prediction results including:
            - disease: Final predicted disease
            - confidence: Final confidence (0-1)
            - classical_prediction: RF + XGBoost results
            - quantum_prediction: QSVM results (if available)
            - ensemble_method: Which method was used
            - models_used: List of models in ensemble
        """
        # Validate symptoms
        validation = self.classical_predictor.validate_symptoms(symptoms)
        valid_symptoms = validation['valid_symptoms']
        
        if not valid_symptoms:
            return {
                'disease': 'Unable to diagnose',
                'confidence': 0.0,
                'classical_prediction': None,
                'quantum_prediction': None,
                'ensemble_method': 'none',
                'error': 'No valid symptoms provided',
                'valid_symptoms': [],
                'invalid_symptoms': validation['invalid_symptoms']
            }
        
        # Get classical predictions (RF + XGBoost)
        classical_result = self.classical_predictor.predict_with_both_models(valid_symptoms)
        
        rf_disease = classical_result['random_forest']['predicted_disease']
        rf_confidence = classical_result['random_forest']['confidence_percentage']
        xgb_disease = classical_result['xgboost']['predicted_disease']
        xgb_confidence = classical_result['xgboost']['confidence_percentage']
        
        # If quantum not available or not requested, use classical only
        if not use_quantum or not self.qsvm_available:
            # Classical ensemble (RF + XGBoost average)
            if rf_disease == xgb_disease:
                final_disease = rf_disease
                final_confidence = (rf_confidence + xgb_confidence) / 2
            else:
                # Models disagree - use higher confidence
                if rf_confidence >= xgb_confidence:
                    final_disease = rf_disease
                    final_confidence = rf_confidence
                else:
                    final_disease = xgb_disease
                    final_confidence = xgb_confidence
            
            return {
                'disease': final_disease,
                'confidence': final_confidence / 100.0,  # Normalize to 0-1
                'classical_prediction': {
                    'random_forest': {'disease': rf_disease, 'confidence': rf_confidence},
                    'xgboost': {'disease': xgb_disease, 'confidence': xgb_confidence},
                    'agreement': rf_disease == xgb_disease
                },
                'quantum_prediction': None,
                'ensemble_method': 'classical_only',
                'models_used': ['Random Forest', 'XGBoost'],
                'valid_symptoms': valid_symptoms,
                'invalid_symptoms': validation['invalid_symptoms']
            }
        
        # Get quantum prediction (QSVM)
        try:
            print("⚛️  Running quantum QSVM prediction...")
            quantum_result = self._predict_quantum(valid_symptoms)
            qsvm_disease = quantum_result['disease']
            qsvm_confidence = quantum_result['confidence']
            
            # Hybrid ensemble voting with weights
            final_disease, final_confidence, voting_details = self._weighted_ensemble_vote(
                rf_disease, rf_confidence,
                xgb_disease, xgb_confidence,
                qsvm_disease, qsvm_confidence
            )
            
            return {
                'disease': final_disease,
                'confidence': final_confidence / 100.0,  # Normalize to 0-1
                'classical_prediction': {
                    'random_forest': {'disease': rf_disease, 'confidence': rf_confidence},
                    'xgboost': {'disease': xgb_disease, 'confidence': xgb_confidence},
                    'agreement': rf_disease == xgb_disease
                },
                'quantum_prediction': {
                    'qsvm': {'disease': qsvm_disease, 'confidence': qsvm_confidence}
                },
                'ensemble_method': 'hybrid_classical_quantum',
                'models_used': ['Random Forest', 'XGBoost', 'QSVM'],
                'weights': {
                    'classical': self.classical_weight,
                    'quantum': self.quantum_weight
                },
                'voting_details': voting_details,
                'valid_symptoms': valid_symptoms,
                'invalid_symptoms': validation['invalid_symptoms']
            }
            
        except Exception as e:
            print(f"⚠️  Quantum prediction failed: {e}")
            print("   Falling back to classical ensemble")
            
            # Fall back to classical only
            if rf_disease == xgb_disease:
                final_disease = rf_disease
                final_confidence = (rf_confidence + xgb_confidence) / 2
            else:
                if rf_confidence >= xgb_confidence:
                    final_disease = rf_disease
                    final_confidence = rf_confidence
                else:
                    final_disease = xgb_disease
                    final_confidence = xgb_confidence
            
            return {
                'disease': final_disease,
                'confidence': final_confidence / 100.0,
                'classical_prediction': {
                    'random_forest': {'disease': rf_disease, 'confidence': rf_confidence},
                    'xgboost': {'disease': xgb_disease, 'confidence': xgb_confidence},
                    'agreement': rf_disease == xgb_disease
                },
                'quantum_prediction': {'error': str(e)},
                'ensemble_method': 'classical_fallback',
                'models_used': ['Random Forest', 'XGBoost'],
                'valid_symptoms': valid_symptoms,
                'invalid_symptoms': validation['invalid_symptoms']
            }
    
    def _predict_quantum(self, symptoms: List[str]) -> Dict:
        """Get quantum QSVM prediction"""
        # Create feature vector
        feature_vector = self.classical_predictor._create_feature_vector(symptoms)
        feature_vector = feature_vector.reshape(1, -1)
        
        # Predict with QSVM
        prediction = self.quantum_predictor.predict(feature_vector)
        disease = prediction[0]
        
        # Get confidence from decision function
        try:
            # Encode features for quantum
            X_quantum = self.quantum_predictor.encoder.encode_features(feature_vector)
            
            # Compute kernel
            kernel_matrix = self.quantum_predictor.quantum_kernel.compute_kernel_matrix(
                self.quantum_predictor.X_train_quantum,
                X_quantum
            )
            
            # Get decision scores
            decision = self.quantum_predictor.svm.decision_function(kernel_matrix.T)[0]
            
            # Convert to confidence percentage (0-100)
            # Use sigmoid-like transformation
            if isinstance(decision, np.ndarray):
                decision = np.max(decision)
            confidence = (1 / (1 + np.exp(-decision))) * 100
            confidence = min(100.0, max(0.0, confidence))
            
        except Exception as e:
            print(f"   Warning: Could not calculate quantum confidence: {e}")
            confidence = 50.0  # Default confidence
        
        return {
            'disease': disease,
            'confidence': confidence
        }
    
    def _weighted_ensemble_vote(
        self, 
        rf_disease: str, rf_conf: float,
        xgb_disease: str, xgb_conf: float,
        qsvm_disease: str, qsvm_conf: float
    ) -> Tuple[str, float, Dict]:
        """
        Weighted ensemble voting combining classical and quantum predictions
        
        Strategy:
        1. Calculate classical consensus (RF + XGBoost)
        2. Apply weights: Classical 70%, Quantum 30%
        3. Final decision based on weighted voting
        
        Returns:
            (final_disease, final_confidence, voting_details)
        """
        # Classical consensus
        classical_agree = (rf_disease == xgb_disease)
        
        if classical_agree:
            classical_disease = rf_disease
            classical_confidence = (rf_conf + xgb_conf) / 2
        else:
            # Use higher confidence classical model
            if rf_conf >= xgb_conf:
                classical_disease = rf_disease
                classical_confidence = rf_conf
            else:
                classical_disease = xgb_disease
                classical_confidence = xgb_conf
        
        # Apply ensemble weights
        weighted_classical = classical_confidence * self.classical_weight
        weighted_quantum = qsvm_conf * self.quantum_weight
        
        # Voting details for transparency
        voting_details = {
            'classical_disease': classical_disease,
            'classical_confidence': classical_confidence,
            'classical_agreement': classical_agree,
            'quantum_disease': qsvm_disease,
            'quantum_confidence': qsvm_conf,
            'weighted_classical_score': weighted_classical,
            'weighted_quantum_score': weighted_quantum
        }
        
        # Final decision
        if classical_disease == qsvm_disease:
            # All models agree
            final_disease = classical_disease
            final_confidence = weighted_classical + weighted_quantum
            voting_details['decision_reason'] = 'unanimous_agreement'
        else:
            # Models disagree - use weighted vote
            if weighted_classical >= weighted_quantum:
                final_disease = classical_disease
                final_confidence = weighted_classical
                voting_details['decision_reason'] = 'classical_higher_weighted_score'
            else:
                final_disease = qsvm_disease
                final_confidence = weighted_quantum
                voting_details['decision_reason'] = 'quantum_higher_weighted_score'
        
        # Normalize confidence to 0-100 range
        final_confidence = min(100.0, max(0.0, final_confidence))
        
        return final_disease, final_confidence, voting_details
    
    def predict(self, symptoms: List[str], use_quantum=True):
        """
        Main prediction method - calls hybrid ensemble
        
        Args:
            symptoms: List of symptom names
            use_quantum: Whether to use quantum model
            
        Returns:
            Hybrid prediction results
        """
        return self.predict_hybrid(symptoms, use_quantum=use_quantum)
    
    def predict_with_ensemble(self, symptoms: List[str]):
        """
        Predict using both RF and XGBoost (classical ensemble only)
        
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
    
    def get_available_symptoms(self) -> List[str]:
        """Get list of all available symptoms"""
        return self.classical_predictor.get_available_symptoms()
    
    def get_available_diseases(self) -> List[str]:
        """Get list of all available diseases"""
        return self.classical_predictor.get_available_diseases()
    
    def is_quantum_available(self) -> bool:
        """Check if quantum model is available"""
        return self.qsvm_available
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'classical_models': {
                'random_forest': 'loaded',
                'xgboost': 'loaded'
            },
            'quantum_models': {
                'qsvm': 'loaded' if self.qsvm_available else 'not_available'
            },
            'ensemble_weights': {
                'classical': self.classical_weight,
                'quantum': self.quantum_weight
            },
            'available_symptoms': len(self.get_available_symptoms()),
            'available_diseases': len(self.get_available_diseases())
        }


def demo_hybrid_prediction():
    """
    Demonstrate hybrid quantum-classical ensemble prediction
    """
    print("\n" + "="*70)
    print("HYBRID CLASSICAL-QUANTUM ENSEMBLE DEMO")
    print("="*70)
    
    # Initialize hybrid predictor
    hybrid = HybridPredictor()
    
    # Model info
    info = hybrid.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"   • Classical Models: {', '.join([k.upper() for k,v in info['classical_models'].items() if v == 'loaded'])}")
    print(f"   • Quantum Models: {', '.join([k.upper() for k,v in info['quantum_models'].items() if v == 'loaded']) or 'None'}")
    print(f"   • Total Symptoms: {info['available_symptoms']}")
    print(f"   • Total Diseases: {info['available_diseases']}")
    
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
    print("TESTING HYBRID ENSEMBLE PREDICTIONS")
    print("="*70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"{test_case['name']}")
        print(f"{'='*70}")
        print(f"Symptoms: {', '.join(test_case['symptoms'])}")
        
        # Hybrid prediction
        start = time.time()
        result = hybrid.predict_hybrid(test_case['symptoms'], use_quantum=True)
        prediction_time = time.time() - start
        
        print(f"\n🎯 HYBRID ENSEMBLE PREDICTION:")
        print(f"   • Disease: {result['disease']}")
        print(f"   • Confidence: {result['confidence']*100:.2f}%")
        print(f"   • Method: {result['ensemble_method']}")
        print(f"   • Models Used: {', '.join(result['models_used'])}")
        print(f"   • Time: {prediction_time:.4f}s")
        
        if result['classical_prediction']:
            print(f"\n� Classical Models:")
            cp = result['classical_prediction']
            rf = cp['random_forest']
            xgb = cp['xgboost']
            print(f"   • Random Forest: {rf['disease']} ({rf['confidence']:.2f}%)")
            print(f"   • XGBoost: {xgb['disease']} ({xgb['confidence']:.2f}%)")
            print(f"   • Agreement: {'✅ Yes' if cp['agreement'] else '❌ No'}")
        
        if result['quantum_prediction'] and 'qsvm' in result['quantum_prediction']:
            print(f"\n⚛️  Quantum Model:")
            qp = result['quantum_prediction']['qsvm']
            print(f"   • QSVM: {qp['disease']} ({qp['confidence']:.2f}%)")
        
        if 'weights' in result:
            print(f"\n⚖️  Ensemble Weights:")
            print(f"   • Classical: {result['weights']['classical']*100:.0f}%")
            print(f"   • Quantum: {result['weights']['quantum']*100:.0f}%")
        
        if 'voting_details' in result:
            vd = result['voting_details']
            print(f"\n🗳️  Voting Details:")
            print(f"   • Classical Score: {vd['weighted_classical_score']:.2f}")
            print(f"   • Quantum Score: {vd['weighted_quantum_score']:.2f}")
            print(f"   • Decision: {vd['decision_reason'].replace('_', ' ').title()}")
    
    print(f"\n{'='*70}")
    print("✅ HYBRID ENSEMBLE DEMO COMPLETE")
    print(f"{'='*70}")
    print(f"\n💡 Summary:")
    print(f"   • Classical Models: ✅ RF + XGBoost (70% weight)")
    print(f"   • Quantum Model: {'✅ QSVM (30% weight)' if hybrid.is_quantum_available() else '❌ Not loaded'}")
    print(f"   • Ensemble Strategy: Weighted voting with fallback")
    print(f"   • Best of Both Worlds: Classical accuracy + Quantum complexity")


if __name__ == "__main__":
    demo_hybrid_prediction()
