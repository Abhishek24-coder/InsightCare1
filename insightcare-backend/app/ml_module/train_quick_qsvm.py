"""
Quick QSVM Training - Smaller Subset for Fast Demo
Trains QSVM on 3 diseases instead of 8 for quick demonstration
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import time
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineering
from quantum_circuit import QuantumFeatureEncoder, QuantumKernel


def train_quick_qsvm():
    """
    Train QSVM quickly on 3 diseases (instead of 8)
    """
    print("\n" + "="*70)
    print("QUICK QSVM TRAINING - 3 DISEASES FOR DEMO")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data...")
    pipeline = DataPipeline()
    pipeline.load_data()
    df = pipeline.prepare_data()
    print(f"✓ Loaded {len(df)} records")
    
    # Feature engineering
    print("\n[2/5] Preparing features...")
    
    # Load pre-trained models to get features
    models_dir = Path(__file__).parent / "models"
    with open(models_dir / "label_encoder.pkl", 'rb') as f:
        encoder_data = pickle.load(f)
    
    symptoms_list = encoder_data['symptoms_list']
    severity_dict = encoder_data['severity_dict']
    
    # Create feature vectors manually
    X_list = []
    y_list = []
    
    for _, row in df.iterrows():
        # Get symptoms for this patient
        patient_symptoms = []
        for col in df.columns:
            if col != 'Disease' and pd.notna(row[col]) and row[col] != '':
                symptom = row[col].strip().replace('_', ' ')
                if symptom:
                    patient_symptoms.append(symptom)
        
        # Create feature vector with severity weights
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
    
    print(f"✓ Features: {X.shape}")
    
    # Select only 3 diseases for quick training
    selected_diseases = [
        'Diabetes',
        'Malaria',
        'Pneumonia'
    ]
    
    print(f"\n[3/5] Selecting {len(selected_diseases)} diseases for quick demo...")
    for i, disease in enumerate(selected_diseases, 1):
        print(f"  {i}. {disease}")
    
    # Filter dataset
    mask = np.isin(y, selected_diseases)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"\n✓ Filtered dataset: {X_filtered.shape[0]} samples")
    
    # Split data (smaller test set for speed)
    print("\n[4/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, 
        test_size=0.15,  # Smaller test set
        random_state=42,
        stratify=y_filtered
    )
    print(f"✓ Train: {len(X_train)} samples")
    print(f"✓ Test: {len(X_test)} samples")
    
    # Train QSVM with smaller configuration
    print("\n[5/5] Training QSVM (quick config)...")
    print("="*70)
    
    # Initialize quantum encoder
    print("✓ Initializing quantum encoder...")
    encoder = QuantumFeatureEncoder(
        n_features=131,
        encoding_type='zz',
        n_qubits=8  # Reduced from 10 for speed
    )
    
    # Create feature map with fewer reps
    feature_map = encoder.create_feature_map(reps=1)  # Reduced from 2
    
    # Initialize quantum kernel
    print("✓ Creating quantum kernel...")
    quantum_kernel = QuantumKernel(feature_map)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"✓ Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"✓ Number of classes: {len(np.unique(y_train_encoded))}")
    
    # Reduce features for quantum encoding
    X_train_quantum = encoder.encode_features(X_train)
    X_test_quantum = encoder.encode_features(X_test)
    print(f"✓ Quantum features: {X_train_quantum.shape}")
    
    # Compute quantum kernel matrix
    print(f"\n⚛️  Computing quantum kernel matrix...")
    print(f"   Training samples: {len(X_train_quantum)}")
    print(f"   This should take 2-5 minutes...")
    
    start_time = time.time()
    kernel_train = quantum_kernel.compute_kernel_matrix(X_train_quantum)
    kernel_time = time.time() - start_time
    
    print(f"✓ Training kernel computed in {kernel_time:.2f}s")
    print(f"  • Matrix shape: {kernel_train.shape}")
    
    # Train SVM with precomputed quantum kernel
    print(f"\n🔧 Training SVM with quantum kernel...")
    svm = SVC(kernel='precomputed', C=1.0, decision_function_shape='ovr')
    svm.fit(kernel_train, y_train_encoded)
    
    # Training accuracy
    y_train_pred = svm.predict(kernel_train)
    train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
    
    print(f"✓ Training accuracy: {train_accuracy*100:.2f}%")
    print(f"  • Support vectors: {len(svm.support_)}")
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    kernel_test = quantum_kernel.compute_kernel_matrix(X_train_quantum, X_test_quantum)
    
    y_test_pred = svm.predict(kernel_test.T)
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
    
    print(f"✓ Test accuracy: {test_accuracy*100:.2f}%")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, 
        label_encoder.inverse_transform(y_test_pred),
        zero_division=0
    ))
    
    # Save model
    print(f"\n💾 Saving QSVM model...")
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_data = {
        'svm': svm,
        'label_encoder': label_encoder,
        'X_train_quantum': X_train_quantum,
        'encoder': encoder,
        'feature_map': feature_map,
        'quantum_kernel': quantum_kernel,
        'n_qubits': 8,
        'encoding_type': 'zz',
        'reps': 1,
        'diseases': selected_diseases
    }
    
    model_path = models_dir / "qsvm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ QSVM model saved to {model_path}")
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("✅ QUICK QSVM TRAINING COMPLETE")
    print("="*70)
    print(f"\n📊 Results:")
    print(f"  • Training accuracy: {train_accuracy*100:.2f}%")
    print(f"  • Test accuracy: {test_accuracy*100:.2f}%")
    print(f"  • Training time: {total_time:.2f}s")
    print(f"  • Diseases: {len(selected_diseases)} (Diabetes, Malaria, Pneumonia)")
    print(f"  • Samples: {len(X_train)} train, {len(X_test)} test")
    print(f"  • Quantum encoding: ZZ Feature Map (8 qubits, 1 rep)")
    print(f"  • Model saved: ✅")
    print("\n💡 Note: This is a quick demo model. For production, train on all diseases.")
    
    return model_data


if __name__ == "__main__":
    train_quick_qsvm()
