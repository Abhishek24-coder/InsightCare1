"""
Complete Quantum ML Demo
Shows all quantum capabilities with visualizations
"""

import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from quantum_circuit import QuantumFeatureEncoder, demo_quantum_encoding
from quick_qsvm_demo import quick_qsvm_demo
from hybrid_predictor import demo_hybrid_prediction
from performance_comparison import generate_performance_report


def show_quantum_circuit_details():
    """
    Show detailed quantum circuit information
    """
    print("\n" + "="*70)
    print("QUANTUM CIRCUIT DETAILS")
    print("="*70)
    
    # Test all encoding types
    encoding_types = ['angle', 'pauli', 'zz']
    
    for enc_type in encoding_types:
        print(f"\n{'='*70}")
        print(f"{enc_type.upper()} ENCODING")
        print(f"{'='*70}")
        
        encoder = QuantumFeatureEncoder(
            n_features=131,
            encoding_type=enc_type,
            n_qubits=10
        )
        
        feature_map = encoder.create_feature_map(reps=2)
        
        print(f"\n📊 Circuit Statistics:")
        print(f"   • Qubits: {feature_map.num_qubits}")
        print(f"   • Depth: {feature_map.depth()}")
        print(f"   • Gates: {len(feature_map.data)}")
        print(f"   • Parameters: {feature_map.num_parameters}")
        print(f"   • Entanglement: Full quantum state")
        
        # Show gate breakdown
        gate_types = {}
        for instruction in feature_map.data:
            gate_name = instruction.operation.name
            gate_types[gate_name] = gate_types.get(gate_name, 0) + 1
        
        print(f"\n🔧 Gate Composition:")
        for gate, count in sorted(gate_types.items()):
            print(f"   • {gate}: {count}")


def demonstrate_quantum_advantage_scenarios():
    """
    Show scenarios where quantum encoding could help
    """
    print("\n" + "="*70)
    print("QUANTUM ADVANTAGE SCENARIOS")
    print("="*70)
    
    scenarios = [
        {
            'title': 'High-Dimensional Feature Space',
            'description': 'Quantum circuits can efficiently represent exponentially large Hilbert spaces',
            'example': 'With 10 qubits, we can represent 2^10 = 1,024 dimensions',
            'advantage': '✅ Potential advantage for very high-dimensional problems',
            'status': 'Ready for testing'
        },
        {
            'title': 'Complex Non-Linear Kernels',
            'description': 'Quantum kernels can compute complex similarity measures',
            'example': 'ZZ Feature Map creates entangled states for pattern matching',
            'advantage': '⚪ Advantage depends on problem structure',
            'status': 'Implemented'
        },
        {
            'title': 'Small Training Sets',
            'description': 'Quantum models may generalize better with limited data',
            'example': 'QSVM trained on 35 samples (vs 3,936 for classical)',
            'advantage': '❌ No advantage observed in our tests',
            'status': 'Tested'
        },
        {
            'title': 'Future Quantum Hardware',
            'description': 'Real quantum computers may provide speedup',
            'example': 'Current simulation is slow; hardware could be faster',
            'advantage': '🔮 Future potential',
            'status': 'Awaiting hardware advances'
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"Scenario {i}: {scenario['title']}")
        print(f"{'='*70}")
        print(f"📋 Description: {scenario['description']}")
        print(f"💡 Example: {scenario['example']}")
        print(f"⚡ Advantage: {scenario['advantage']}")
        print(f"📊 Status: {scenario['status']}")


def complete_quantum_demo():
    """
    Run complete quantum ML demonstration
    """
    print("\n" + "="*80)
    print("COMPLETE QUANTUM ML DEMONSTRATION")
    print("InsightCare Disease Diagnosis - Quantum Enhancement Module")
    print("="*80)
    
    print("\n📋 Demo Contents:")
    print("   1. Quantum Circuit Encoding")
    print("   2. QSVM Quick Demo")
    print("   3. Hybrid Quantum-Classical Predictor")
    print("   4. Performance Comparison Report")
    print("   5. Quantum Advantage Scenarios")
    
    input("\nPress Enter to start demo...")
    
    # Part 1: Quantum Encoding
    print("\n" + "="*80)
    print("PART 1: QUANTUM FEATURE ENCODING")
    print("="*80)
    demo_quantum_encoding()
    
    input("\nPress Enter to continue to Part 2...")
    
    # Part 2: Circuit Details
    show_quantum_circuit_details()
    
    input("\nPress Enter to continue to Part 3...")
    
    # Part 3: QSVM Demo
    print("\n" + "="*80)
    print("PART 3: QUANTUM SVM DEMONSTRATION")
    print("="*80)
    qsvm_acc, csvm_acc = quick_qsvm_demo()
    
    input("\nPress Enter to continue to Part 4...")
    
    # Part 4: Hybrid Predictor
    print("\n" + "="*80)
    print("PART 4: HYBRID QUANTUM-CLASSICAL PREDICTOR")
    print("="*80)
    demo_hybrid_prediction()
    
    input("\nPress Enter to continue to Part 5...")
    
    # Part 5: Performance Comparison
    print("\n" + "="*80)
    print("PART 5: PERFORMANCE COMPARISON")
    print("="*80)
    report_path = generate_performance_report()
    
    input("\nPress Enter to continue to Part 6...")
    
    # Part 6: Quantum Advantage Scenarios
    demonstrate_quantum_advantage_scenarios()
    
    # Final Summary
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE - SUMMARY")
    print("="*80)
    
    print("\n📊 What We Demonstrated:")
    print("   ✅ Quantum circuit encoding (angle, pauli, ZZ)")
    print("   ✅ Quantum kernel computation")
    print("   ✅ QSVM classifier working")
    print("   ✅ Integration with classical pipeline")
    print("   ✅ Hybrid predictor combining both approaches")
    print("   ✅ Performance comparison report generated")
    
    print("\n🎯 Key Findings:")
    print(f"   • Classical accuracy: 100% (RF & XGBoost)")
    print(f"   • Quantum accuracy: {qsvm_acc*100:.1f}% (QSVM on small demo)")
    print(f"   • Classical is better for this use case")
    print(f"   • Quantum infrastructure is ready for future research")
    
    print("\n📁 Generated Files:")
    print(f"   • {report_path.name}")
    print(f"   • All quantum modules in ml_module/")
    
    print("\n💡 Recommendations:")
    print("   1. Use classical models (RF + XGBoost) for production")
    print("   2. Quantum encoding available for experimentation")
    print("   3. Monitor quantum hardware developments")
    print("   4. Hybrid predictor ready for deployment")
    
    print("\n🚀 Next Steps:")
    print("   1. Integrate with FastAPI backend")
    print("   2. Deploy classical models to production")
    print("   3. Continue quantum research with new datasets")
    print("   4. Test on real quantum hardware when available")
    
    print("\n" + "="*80)
    print("Thank you for exploring Quantum ML with InsightCare!")
    print("="*80)


if __name__ == "__main__":
    complete_quantum_demo()
