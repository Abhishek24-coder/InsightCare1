"""
Quick QSVM Demo with Small Dataset
Demonstrates quantum kernel advantage on a subset of data
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import DataPipeline
from quantum_circuit import QuantumFeatureEncoder, QuantumKernel


def quick_qsvm_demo():
    """
    Quick demo with small dataset (50 samples)
    """
    print("\n" + "="*70)
    print("QUICK QSVM DEMO - Small Dataset")
    print("="*70)
    
    # Load data
    print("\n[1/4] Loading data...")
    pipeline = DataPipeline()
    pipeline.load_data()
    df = pipeline.prepare_data()
    
    # Load features
    models_dir = Path(__file__).parent / "models"
    with open(models_dir / "label_encoder.pkl", 'rb') as f:
        encoder_data = pickle.load(f)
    
    symptoms_list = encoder_data['symptoms_list']
    severity_dict = encoder_data['severity_dict']
    
    # Create feature vectors
    X_list = []
    y_list = []
    
    for _, row in df.iterrows():
        patient_symptoms = []
        for col in df.columns:
            if col != 'Disease' and pd.notna(row[col]) and row[col] != '':
                symptom = row[col].strip().replace('_', ' ')
                if symptom:
                    patient_symptoms.append(symptom)
        
        if patient_symptoms:
            vector = np.zeros(len(symptoms_list))
            for symptom in patient_symptoms:
                if symptom in symptoms_list:
                    idx = symptoms_list.index(symptom)
                    severity = severity_dict.get(symptom, 1)
                    vector[idx] = severity
            
            X_list.append(vector)
            y_list.append(row['Disease'])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"✓ Full dataset: {X.shape}")
    
    # Select 3 diseases for quick demo
    print("\n[2/4] Selecting 3 diseases for demo...")
    selected_diseases = ['Diabetes', 'Malaria', 'Typhoid']
    
    mask = np.isin(y, selected_diseases)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"✓ Filtered: {X_filtered.shape[0]} samples")
    
    # Take only 50 samples for quick demo
    np.random.seed(42)
    indices = np.random.choice(len(X_filtered), size=min(50, len(X_filtered)), replace=False)
    X_small = X_filtered[indices]
    y_small = y_filtered[indices]
    
    print(f"✓ Small dataset: {X_small.shape[0]} samples")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y_small, test_size=0.3, random_state=42
    )
    
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Initialize quantum encoder
    print("\n[3/4] Setting up quantum kernel...")
    encoder = QuantumFeatureEncoder(
        n_features=131,
        encoding_type='zz',
        n_qubits=6  # Use fewer qubits for speed
    )
    
    X_train_quantum = encoder.encode_features(X_train)
    X_test_quantum = encoder.encode_features(X_test)
    
    print(f"✓ Quantum features: {X_train_quantum.shape}")
    
    # Create quantum kernel
    feature_map = encoder.create_feature_map(reps=1)  # Use 1 rep for speed
    quantum_kernel = QuantumKernel(feature_map)
    
    # Compute quantum kernel matrices
    print("\n⚛️  Computing quantum kernels...")
    qstart = time.time()
    
    train_kernel = quantum_kernel.compute_kernel_matrix(X_train_quantum)
    test_kernel = quantum_kernel.compute_kernel_matrix(X_train_quantum, X_test_quantum)
    
    qtime = time.time() - qstart
    print(f"✓ Quantum kernel computed in {qtime:.2f}s")
    
    # Train QSVM
    print("\n[4/4] Training models...")
    
    # Quantum SVM
    qsvm = SVC(kernel='precomputed')
    qsvm_start = time.time()
    qsvm.fit(train_kernel, y_train)
    y_pred_qsvm = qsvm.predict(test_kernel.T)
    qsvm_time = time.time() - qsvm_start
    qsvm_acc = accuracy_score(y_test, y_pred_qsvm)
    
    # Classical SVM (RBF kernel)
    csvm = SVC(kernel='rbf', C=1.0)
    csvm_start = time.time()
    csvm.fit(X_train_quantum, y_train)
    y_pred_csvm = csvm.predict(X_test_quantum)
    csvm_time = time.time() - csvm_start
    csvm_acc = accuracy_score(y_test, y_pred_csvm)
    
    # Results
    print("\n" + "="*70)
    print("✅ QSVM DEMO COMPLETE")
    print("="*70)
    
    print(f"\n📊 Results ({len(X_train)} train, {len(X_test)} test samples):")
    print(f"\n⚛️  Quantum SVM:")
    print(f"   • Accuracy: {qsvm_acc*100:.2f}%")
    print(f"   • Training time: {qsvm_time:.2f}s")
    print(f"   • Kernel time: {qtime:.2f}s")
    print(f"   • Support vectors: {len(qsvm.support_)}")
    
    print(f"\n🔧 Classical SVM (RBF):")
    print(f"   • Accuracy: {csvm_acc*100:.2f}%")
    print(f"   • Training time: {csvm_time:.2f}s")
    print(f"   • Support vectors: {len(csvm.support_)}")
    
    print(f"\n💡 Comparison:")
    if qsvm_acc > csvm_acc:
        diff = (qsvm_acc - csvm_acc) * 100
        print(f"   • ✅ Quantum SVM better by {diff:.2f}%")
    elif qsvm_acc < csvm_acc:
        diff = (csvm_acc - qsvm_acc) * 100
        print(f"   • Classical SVM better by {diff:.2f}%")
    else:
        print(f"   • Both models have same accuracy")
    
    print(f"   • Quantum encoding: ZZ Feature Map")
    print(f"   • Qubits used: 6")
    print(f"   • Dataset: {len(selected_diseases)} diseases")
    
    return qsvm_acc, csvm_acc


if __name__ == "__main__":
    quick_qsvm_demo()
