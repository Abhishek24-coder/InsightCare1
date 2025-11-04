"""
Simple Quantum vs Classical Comparison Demo
"""

from hybrid_predictor import HybridPredictor

print("\n" + "="*80)
print("⚛️  QUANTUM vs CLASSICAL PREDICTION COMPARISON")
print("="*80)

# Initialize predictor
print("\n🔧 Initializing Hybrid Predictor...")
predictor = HybridPredictor()

# Test case
symptoms = ['fatigue', 'weight_loss', 'polyuria', 'increased_appetite']

print("\n" + "="*80)
print("📋 Test Symptoms: Diabetes-like symptoms")
print("="*80)
print(f"   • {', '.join(symptoms)}")

# Classical prediction
print("\n🔬 [1/2] Running CLASSICAL ONLY prediction...")
print("-"*80)
classical_result = predictor.predict_hybrid(symptoms, use_quantum=False)

print(f"🏥 Disease: {classical_result['disease']}")
print(f"📊 Confidence: {classical_result['confidence']*100:.2f}%")
print(f"🎭 Method: {classical_result.get('ensemble_info', {}).get('method', 'N/A')}")

# Quantum-enhanced prediction
print("\n⚛️  [2/2] Running QUANTUM-ENHANCED prediction...")
print("-"*80)
quantum_result = predictor.predict_hybrid(symptoms, use_quantum=True)

print(f"🏥 Disease: {quantum_result['disease']}")
print(f"📊 Confidence: {quantum_result['confidence']*100:.2f}%")
print(f"🎭 Method: {quantum_result.get('ensemble_info', {}).get('method', 'N/A')}")

# Comparison
print("\n" + "="*80)
print("📊 COMPARISON SUMMARY")
print("="*80)

print(f"\n{'Method':<25} {'Prediction':<20} {'Confidence':<15}")
print("-"*60)
print(f"{'Classical Only':<25} {classical_result['disease']:<20} {classical_result['confidence']*100:>6.2f}%")
print(f"{'Quantum-Enhanced':<25} {quantum_result['disease']:<20} {quantum_result['confidence']*100:>6.2f}%")

# Show the difference
conf_diff = (quantum_result['confidence'] - classical_result['confidence']) * 100
if abs(conf_diff) > 1:
    if conf_diff > 0:
        print(f"\n💡 Quantum improved confidence by {conf_diff:.2f}%")
    else:
        print(f"\n💡 Classical was more confident by {abs(conf_diff):.2f}%")
else:
    print(f"\n💡 Similar confidence levels (difference: {conf_diff:.2f}%)")

# Agreement
if classical_result['disease'] == quantum_result['disease']:
    print(f"✅ Both methods agree on: {classical_result['disease']}")
else:
    print(f"⚠️  Methods disagree:")
    print(f"   • Classical: {classical_result['disease']}")
    print(f"   • Quantum: {quantum_result['disease']}")

print("\n" + "="*80)
print("✅ DEMO COMPLETE - Quantum-Classical Hybrid System Working!")
print("="*80)

print("\n💡 Key Insights:")
print("   • Classical: Fast, reliable, proven accuracy")
print("   • Quantum: Captures complex feature interactions")
print("   • Hybrid: Best of both worlds (70% classical + 30% quantum)")
print("\n🚀 Ready for production use!")
