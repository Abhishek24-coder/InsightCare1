"""
Quantum Support Vector Machine (QSVM) for Disease Diagnosis
Uses quantum kernel for complex disease classification
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


class QSVMClassifier:
    """
    Quantum Support Vector Machine for disease classification
    """
    
    def __init__(self, n_qubits=10, encoding_type='zz', reps=2):
        """
        Initialize QSVM classifier
        
        Args:
            n_qubits: Number of qubits for quantum circuit
            encoding_type: Type of quantum encoding ('angle', 'pauli', 'zz')
            reps: Number of repetitions in quantum circuit
        """
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.reps = reps
        
        # Initialize quantum encoder
        self.encoder = QuantumFeatureEncoder(
            n_features=131,
            encoding_type=encoding_type,
            n_qubits=n_qubits
        )
        
        # Create feature map
        self.feature_map = self.encoder.create_feature_map(reps=reps)
        
        # Initialize quantum kernel
        self.quantum_kernel = QuantumKernel(self.feature_map)
        
        # Classical SVM (will use quantum kernel)
        self.svm = None
        self.label_encoder = LabelEncoder()
        
        print(f"✓ QSVM Classifier initialized")
        print(f"  • Qubits: {n_qubits}")
        print(f"  • Encoding: {encoding_type}")
        print(f"  • Reps: {reps}")
    
    def train(self, X_train, y_train, selected_diseases=None):
        """
        Train QSVM on selected diseases
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels
            selected_diseases: List of diseases to train on (None = all)
            
        Returns:
            Training accuracy and time
        """
        print(f"\n{'='*70}")
        print("TRAINING QSVM MODEL")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Filter by selected diseases if specified
        if selected_diseases is not None:
            mask = np.isin(y_train, selected_diseases)
            X_train = X_train[mask]
            y_train = y_train[mask]
            print(f"✓ Filtered to {len(selected_diseases)} diseases")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        print(f"✓ Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"✓ Number of classes: {len(np.unique(y_train_encoded))}")
        
        # Reduce features for quantum encoding
        X_train_quantum = self.encoder.encode_features(X_train)
        print(f"✓ Quantum features: {X_train_quantum.shape}")
        
        # Compute quantum kernel matrix
        print(f"\n⚛️  Computing quantum kernel matrix...")
        print(f"   This may take a few minutes for large datasets...")
        
        kernel_start = time.time()
        kernel_matrix = self.quantum_kernel.compute_kernel_matrix(X_train_quantum)
        kernel_time = time.time() - kernel_start
        
        print(f"✓ Kernel computed in {kernel_time:.2f}s")
        print(f"  • Matrix shape: {kernel_matrix.shape}")
        print(f"  • Matrix stats: min={kernel_matrix.min():.4f}, max={kernel_matrix.max():.4f}, mean={kernel_matrix.mean():.4f}")
        
        # Train SVM with precomputed quantum kernel
        print(f"\n🔧 Training SVM with quantum kernel...")
        self.svm = SVC(kernel='precomputed', C=1.0, decision_function_shape='ovr')
        self.svm.fit(kernel_matrix, y_train_encoded)
        
        # Training accuracy
        y_pred = self.svm.predict(kernel_matrix)
        train_accuracy = accuracy_score(y_train_encoded, y_pred)
        
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time:.2f}s")
        print(f"  • Training accuracy: {train_accuracy*100:.2f}%")
        print(f"  • Support vectors: {len(self.svm.support_)}")
        
        # Store training data for kernel computation during prediction
        self.X_train_quantum = X_train_quantum
        
        return train_accuracy, training_time
    
    def predict(self, X_test):
        """
        Predict using QSVM
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted labels
        """
        if self.svm is None:
            raise ValueError("Model not trained yet!")
        
        # Reduce features for quantum encoding
        X_test_quantum = self.encoder.encode_features(X_test)
        
        # Compute kernel between test and training data
        kernel_matrix = self.quantum_kernel.compute_kernel_matrix(
            self.X_train_quantum, 
            X_test_quantum
        )
        
        # Predict
        y_pred_encoded = self.svm.predict(kernel_matrix.T)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate QSVM on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Test accuracy
        """
        print(f"\n{'='*70}")
        print("EVALUATING QSVM MODEL")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        eval_time = time.time() - start_time
        
        print(f"\n✓ Evaluation completed in {eval_time:.2f}s")
        print(f"  • Test accuracy: {accuracy*100:.2f}%")
        print(f"  • Test samples: {len(y_test)}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        return accuracy
    
    def save_model(self, filepath):
        """Save QSVM model"""
        model_data = {
            'svm': self.svm,
            'label_encoder': self.label_encoder,
            'X_train_quantum': self.X_train_quantum,
            'n_qubits': self.n_qubits,
            'encoding_type': self.encoding_type,
            'reps': self.reps
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ QSVM model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load QSVM model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Recreate QSVM
        qsvm = cls(
            n_qubits=model_data['n_qubits'],
            encoding_type=model_data['encoding_type'],
            reps=model_data['reps']
        )
        
        qsvm.svm = model_data['svm']
        qsvm.label_encoder = model_data['label_encoder']
        qsvm.X_train_quantum = model_data['X_train_quantum']
        
        print(f"✓ QSVM model loaded from {filepath}")
        
        return qsvm


def train_qsvm_on_complex_diseases():
    """
    Train QSVM on a subset of complex/confusing diseases
    """
    print("\n" + "="*70)
    print("QSVM TRAINING ON COMPLEX DISEASES")
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
    
    # Select complex diseases (diseases with similar symptoms)
    # For demo, let's pick 8 diseases
    selected_diseases = [
        'Diabetes',
        'Hypoglycemia',
        'Hypertension',
        'Hyperthyroidism',
        'Hypothyroidism',
        'Malaria',
        'Typhoid',
        'Dengue'
    ]
    
    print(f"\n[3/5] Selecting {len(selected_diseases)} complex diseases...")
    for i, disease in enumerate(selected_diseases, 1):
        print(f"  {i}. {disease}")
    
    # Filter dataset
    mask = np.isin(y, selected_diseases)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"\n✓ Filtered dataset: {X_filtered.shape[0]} samples")
    
    # Split data
    print("\n[4/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, 
        test_size=0.2, 
        random_state=42,
        stratify=y_filtered
    )
    print(f"✓ Train: {len(X_train)} samples")
    print(f"✓ Test: {len(X_test)} samples")
    
    # Train QSVM
    print("\n[5/5] Training QSVM...")
    qsvm = QSVMClassifier(n_qubits=10, encoding_type='zz', reps=2)
    train_acc, train_time = qsvm.train(X_train, y_train)
    
    # Evaluate
    test_acc = qsvm.evaluate(X_test, y_test)
    
    # Save model
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    qsvm.save_model(models_dir / "qsvm_model.pkl")
    
    # Summary
    print("\n" + "="*70)
    print("✅ QSVM TRAINING COMPLETE")
    print("="*70)
    print(f"\n📊 Results:")
    print(f"  • Training accuracy: {train_acc*100:.2f}%")
    print(f"  • Test accuracy: {test_acc*100:.2f}%")
    print(f"  • Training time: {train_time:.2f}s")
    print(f"  • Diseases: {len(selected_diseases)}")
    print(f"  • Quantum encoding: ZZ Feature Map")
    print(f"  • Qubits: 10")
    
    return qsvm, train_acc, test_acc


if __name__ == "__main__":
    train_qsvm_on_complex_diseases()
