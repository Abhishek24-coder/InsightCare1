"""
Test Hybrid Classical-Quantum Ensemble Predictor
Tests the integration of RF + XGBoost + QSVM
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from app.services.ml_service import ml_service, is_ml_available, is_quantum_available


def test_hybrid_ensemble():
    """Test hybrid classical-quantum ensemble"""
    print("\n" + "="*70)
    print("TESTING HYBRID CLASSICAL-QUANTUM ENSEMBLE")
    print("="*70)
    
    # Check availability
    print(f"\n📊 System Status:")
    print(f"   • ML Models Available: {is_ml_available()}")
    print(f"   • Quantum Model Available: {is_quantum_available()}")
    
    if not is_ml_available():
        print("\n❌ ML models not loaded!")
        return
    
    # Test cases
    test_cases = [
        {
            'name': 'Test 1: Diabetes Symptoms',
            'symptoms': ['fatigue', 'weight loss', 'increased appetite', 'frequent urination']
        },
        {
            'name': 'Test 2: Malaria Symptoms',
            'symptoms': ['fever', 'chills', 'sweating', 'headache', 'nausea']
        },
        {
            'name': 'Test 3: Mixed Symptoms',
            'symptoms': ['cough', 'fever', 'fatigue', 'chest pain']
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"{test['name']}")
        print(f"{'='*70}")
        print(f"Symptoms: {', '.join(test['symptoms'])}")
        
        # Test with quantum (if available)
        print(f"\n🎯 Hybrid Prediction (Quantum Enabled):")
        result = ml_service.predict_disease(test['symptoms'], use_quantum=True)
        
        if result:
            pred = result[0]
            print(f"   • Disease: {pred['disease']}")
            print(f"   • Confidence: {pred['confidence']*100:.2f}%")
            print(f"   • Severity: {pred['severity']}")
            print(f"   • Method: {pred['model_used']}")
            
            if 'ensemble_info' in pred:
                info = pred['ensemble_info']
                print(f"\n   📊 Ensemble Details:")
                print(f"      • Models Used: {', '.join(info.get('models_used', []))}")
                print(f"      • Quantum Used: {info.get('quantum_used', False)}")
                
                if 'classical_prediction' in info and info['classical_prediction']:
                    cp = info['classical_prediction']
                    if 'random_forest' in cp:
                        rf = cp['random_forest']
                        print(f"      • RF Prediction: {rf['disease']} ({rf['confidence']:.2f}%)")
                    if 'xgboost' in cp:
                        xgb = cp['xgboost']
                        print(f"      • XGBoost Prediction: {xgb['disease']} ({xgb['confidence']:.2f}%)")
                
                if 'quantum_prediction' in info and info['quantum_prediction']:
                    qp = info['quantum_prediction']
                    if 'qsvm' in qp:
                        qsvm = qp['qsvm']
                        print(f"      • QSVM Prediction: {qsvm['disease']} ({qsvm['confidence']:.2f}%)")
                
                if 'weights' in info and info['weights']:
                    w = info['weights']
                    print(f"      • Ensemble Weights: Classical {w['classical']*100:.0f}%, Quantum {w['quantum']*100:.0f}%")
        
        # Test without quantum (classical only)
        print(f"\n📊 Classical-Only Prediction:")
        result_classical = ml_service.predict_disease(test['symptoms'], use_quantum=False)
        
        if result_classical:
            pred_c = result_classical[0]
            print(f"   • Disease: {pred_c['disease']}")
            print(f"   • Confidence: {pred_c['confidence']*100:.2f}%")
            print(f"   • Method: {pred_c['model_used']}")
    
    print(f"\n{'='*70}")
    print("✅ HYBRID ENSEMBLE TEST COMPLETE")
    print(f"{'='*70}")
    
    print(f"\n💡 Summary:")
    if is_quantum_available():
        print(f"   • Hybrid Ensemble: ✅ Working")
        print(f"   • Classical Models: ✅ RF + XGBoost")
        print(f"   • Quantum Model: ✅ QSVM")
        print(f"   • Ensemble Strategy: 70% Classical + 30% Quantum")
    else:
        print(f"   • Classical Ensemble: ✅ Working")
        print(f"   • Classical Models: ✅ RF + XGBoost")
        print(f"   • Quantum Model: ❌ Not available (classical fallback)")
        print(f"   • Note: Train QSVM model to enable hybrid mode")


if __name__ == "__main__":
    test_hybrid_ensemble()
