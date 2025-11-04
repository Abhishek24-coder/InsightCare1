"""Quick script to verify QSVM model training completion"""
import pickle
from pathlib import Path

model_path = Path('models/qsvm_model.pkl')

if model_path.exists():
    print("=" * 70)
    print("✅ QSVM TRAINING COMPLETE!")
    print("=" * 70)
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"\n📦 Model Information:")
    print(f"   • File: {model_path}")
    print(f"   • Size: {model_path.stat().st_size / 1024:.2f} KB")
    print(f"   • Type: {type(model_data)}")
    
    if isinstance(model_data, dict):
        print(f"\n🔑 Model Components:")
        for key in model_data.keys():
            print(f"   • {key}: {type(model_data[key]).__name__}")
        
        # Check specific components
        if 'n_qubits' in model_data:
            print(f"\n⚛️  Quantum Configuration:")
            print(f"   • Qubits: {model_data['n_qubits']}")
            print(f"   • Encoding: {model_data.get('encoding_type', 'N/A')}")
            print(f"   • Repetitions: {model_data.get('reps', 'N/A')}")
        
        if 'diseases' in model_data:
            print(f"\n🏥 Diseases Trained:")
            for i, disease in enumerate(model_data['diseases'], 1):
                print(f"   {i}. {disease}")
        
        if 'svm' in model_data:
            svm = model_data['svm']
            print(f"\n📊 SVM Details:")
            print(f"   • Support vectors: {len(svm.support_)}")
            print(f"   • Classes: {len(svm.classes_)}")
    
    print("\n" + "=" * 70)
    print("✅ MODEL READY FOR INTEGRATION")
    print("=" * 70)
    print("\n💡 Next Steps:")
    print("   1. Restart backend server")
    print("   2. QSVM will auto-load into HybridPredictor")
    print("   3. Quantum predictions enabled!")
    print("\n🚀 Ready to use quantum-enhanced diagnosis!")
    
else:
    print("❌ QSVM model not found")
    print(f"   Expected at: {model_path.absolute()}")
