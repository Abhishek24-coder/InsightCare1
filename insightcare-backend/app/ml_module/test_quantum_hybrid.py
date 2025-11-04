"""
Test Quantum-Classical Hybrid Prediction
Shows the difference between classical-only and quantum-enhanced predictions
"""

from hybrid_predictor import HybridPredictor
import json

print("\n" + "="*80)
print("🌟 QUANTUM-CLASSICAL HYBRID ENSEMBLE DEMO")
print("="*80)

# Initialize hybrid predictor
print("\n[1/4] Initializing Hybrid Predictor...")
predictor = HybridPredictor()

# Test cases
test_cases = [
    {
        'name': 'Diabetes Patient',
        'symptoms': ['fatigue', 'weight_loss', 'polyuria', 'increased_appetite'],
        'expected': 'Diabetes'
    },
    {
        'name': 'Malaria Patient', 
        'symptoms': ['high_fever', 'chills', 'sweating', 'headache', 'nausea'],
        'expected': 'Malaria'
    },
    {
        'name': 'Pneumonia Patient',
        'symptoms': ['cough', 'high_fever', 'breathlessness', 'chest_pain'],
        'expected': 'Pneumonia'
    }
]

print("\n[2/4] Running Predictions...\n")

for i, case in enumerate(test_cases, 1):
    print("="*80)
    print(f"TEST CASE {i}: {case['name']}")
    print("="*80)
    print(f"📋 Symptoms: {', '.join(case['symptoms'])}")
    print(f"🎯 Expected: {case['expected']}")
    
    # Predict with quantum enabled
    print("\n🔬 Running Hybrid Prediction (Quantum Enabled)...")
    result = predictor.predict_hybrid(case['symptoms'], use_quantum=True)
    
    print(f"\n🏥 Prediction: {result['disease']}")
    print(f"📊 Confidence: {result['confidence']*100:.2f}%")
    
    # Show ensemble details
    if 'ensemble_info' in result:
        info = result['ensemble_info']
        print(f"\n🎭 Ensemble Method: {info['method']}")
        print(f"   • Models Used: {', '.join(info['models_used'])}")
        print(f"   • Quantum Available: {info.get('quantum_available', False)}")
        print(f"   • Quantum Used: {info.get('quantum_used', False)}")
        
        # Classical predictions
        if 'classical_prediction' in info:
            cp = info['classical_prediction']
            print(f"\n🌲 Classical Models:")
            if 'random_forest' in cp:
                rf = cp['random_forest']
                print(f"   • Random Forest: {rf['disease']} ({rf['confidence']:.2f}%)")
            if 'xgboost' in cp:
                xgb = cp['xgboost']
                print(f"   • XGBoost: {xgb['disease']} ({xgb['confidence']:.2f}%)")
            print(f"   • Agreement: {'✅ YES' if cp.get('agreement', False) else '❌ NO'}")
        
        # Quantum prediction
        if 'quantum_prediction' in info and info.get('quantum_used', False):
            qp = info['quantum_prediction']
            if 'qsvm' in qp:
                qsvm = qp['qsvm']
                print(f"\n⚛️  Quantum Model (QSVM):")
                print(f"   • Prediction: {qsvm['disease']} ({qsvm['confidence']:.2f}%)")
        
        # Voting details
        if 'voting_details' in info:
            vd = info['voting_details']
            print(f"\n🗳️  Weighted Voting:")
            print(f"   • Classical: {vd.get('classical_disease', 'N/A')} "
                  f"(score: {vd.get('weighted_classical_score', 0):.2f})")
            if 'quantum_disease' in vd:
                print(f"   • Quantum: {vd.get('quantum_disease', 'N/A')} "
                      f"(score: {vd.get('weighted_quantum_score', 0):.2f})")
            print(f"   • Decision: {vd.get('decision_reason', 'N/A')}")
    
    # Result
    if result['disease'].lower() == case['expected'].lower():
        print(f"\n✅ CORRECT - Predicted: {result['disease']} (Expected: {case['expected']})")
    else:
        print(f"\n⚠️  MISMATCH - Predicted: {result['disease']} (Expected: {case['expected']})")
    
    print()

print("\n[3/4] Testing Classical-Only Mode (Quantum Disabled)...")
print("="*80)

# Test one case with quantum disabled for comparison
case = test_cases[0]
print(f"📋 Test: {case['name']}")
print(f"   Symptoms: {', '.join(case['symptoms'])}")

result_no_quantum = predictor.predict_hybrid(case['symptoms'], use_quantum=False)

print(f"\n🏥 Prediction (Classical Only): {result_no_quantum['disease']}")
print(f"📊 Confidence: {result_no_quantum['confidence']*100:.2f}%")

if 'ensemble_info' in result_no_quantum:
    info = result_no_quantum['ensemble_info']
    print(f"   • Method: {info['method']}")
    print(f"   • Quantum Used: {info.get('quantum_used', False)}")

print("\n[4/4] Summary")
print("="*80)
print("✅ Hybrid Predictor Test Complete!")
print("\n📊 Capabilities:")
print("   • ✅ Classical ML (Random Forest + XGBoost)")
print("   • ✅ Quantum ML (QSVM)")
print("   • ✅ Hybrid Ensemble (70% Classical + 30% Quantum)")
print("   • ✅ Weighted Voting")
print("   • ✅ Automatic Fallback")

print("\n" + "="*80)
print("🎉 QUANTUM-ENHANCED AI DIAGNOSIS SYSTEM READY!")
print("="*80)
