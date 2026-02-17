"""
Quantum Circuit for Disease Diagnosis
Encodes classical symptom features into quantum states
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
import warnings
warnings.filterwarnings('ignore')


class QuantumFeatureEncoder:
    """
    Encodes classical symptom features into quantum states
    Supports multiple encoding strategies
    """
    
    def __init__(self, n_features=131, encoding_type='angle', n_qubits=None):
        """
        Initialize quantum feature encoder
        
        Args:
            n_features: Number of classical features (131 symptoms)
            encoding_type: 'angle', 'amplitude', or 'pauli'
            n_qubits: Number of qubits (auto-calculated if None)
        """
        self.n_features = n_features
        self.encoding_type = encoding_type
        
        # Calculate optimal number of qubits
        if n_qubits is None:
            if encoding_type == 'amplitude':
                # Amplitude encoding: 2^n qubits can encode 2^n features
                self.n_qubits = int(np.ceil(np.log2(n_features)))
            else:
                # Angle encoding: 1 qubit per feature (we'll use reduced features)
                # Use 10 qubits for reduced feature space
                self.n_qubits = 10
        else:
            self.n_qubits = n_qubits
        
        print(f"✓ Quantum Encoder initialized")
        print(f"  • Features: {self.n_features}")
        print(f"  • Qubits: {self.n_qubits}")
        print(f"  • Encoding: {self.encoding_type}")
    
    def create_feature_map(self, reps=2):
        """
        Create quantum feature map circuit
        
        Args:
            reps: Number of repetitions for the feature map
            
        Returns:
            QuantumCircuit for feature encoding
        """
        if self.encoding_type == 'pauli':
            # Pauli feature map (good for quantum kernels)
            feature_map = PauliFeatureMap(
                feature_dimension=self.n_qubits,
                reps=reps,
                paulis=['Z', 'ZZ'],
                entanglement='full'
            )
            
        elif self.encoding_type == 'zz':
            # ZZ Feature map (default for QSVM)
            feature_map = ZZFeatureMap(
                feature_dimension=self.n_qubits,
                reps=reps,
                entanglement='linear'
            )
            
        elif self.encoding_type == 'angle':
            # Angle encoding (simple rotation-based)
            feature_map = self._create_angle_encoding(reps)
            
        else:
            # Default to ZZ feature map
            feature_map = ZZFeatureMap(
                feature_dimension=self.n_qubits,
                reps=reps,
                entanglement='linear'
            )
        
        return feature_map
    
    def _create_angle_encoding(self, reps=2):
        """
        Create custom angle encoding circuit
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        qc = QuantumCircuit(qr)
        
        # Create parameters for each qubit
        params = [Parameter(f'x[{i}]') for i in range(self.n_qubits)]
        
        for _ in range(reps):
            # Rotation layer
            for i in range(self.n_qubits):
                qc.ry(params[i], qr[i])
            
            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qc.cx(qr[i], qr[i+1])
            
            # Additional rotations
            for i in range(self.n_qubits):
                qc.rz(params[i], qr[i])
        
        return qc
    
    def encode_features(self, features):
        """
        Encode classical features into quantum circuit
        
        Args:
            features: numpy array of shape (n_samples, n_features)
            
        Returns:
            Reduced features ready for quantum encoding
        """
        # Reduce features to n_qubits dimensions using PCA or selection
        if features.shape[1] > self.n_qubits:
            # Simple feature reduction: take features with highest variance
            feature_variance = np.var(features, axis=0)
            top_indices = np.argsort(feature_variance)[-self.n_qubits:]
            reduced_features = features[:, top_indices]
        else:
            reduced_features = features
        
        # Normalize to [0, 2π] for angle encoding
        if self.encoding_type == 'angle':
            min_val = reduced_features.min(axis=0)
            max_val = reduced_features.max(axis=0)
            # Avoid division by zero
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            reduced_features = (reduced_features - min_val) / range_val * 2 * np.pi
        
        return reduced_features
    
    def visualize_circuit(self, save_path=None):
        """
        Visualize the quantum circuit
        
        Args:
            save_path: Path to save circuit diagram (optional)
        """
        feature_map = self.create_feature_map(reps=1)
        
        print("\n" + "="*70)
        print("QUANTUM FEATURE ENCODING CIRCUIT")
        print("="*70)
        print(f"\nCircuit depth: {feature_map.depth()}")
        print(f"Number of qubits: {feature_map.num_qubits}")
        print(f"Number of parameters: {feature_map.num_parameters}")
        print(f"Number of gates: {len(feature_map.data)}")
        
        # Try to draw circuit
        try:
            print("\nCircuit Diagram:")
            print(feature_map.draw(output='text'))
        except Exception as e:
            print(f"Could not draw circuit: {e}")
        
        if save_path:
            try:
                figure = feature_map.draw(output='mpl')
                figure.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"\n✓ Circuit diagram saved to {save_path}")
            except Exception as e:
                print(f"Could not save circuit diagram: {e}")
        
        return feature_map


class QuantumKernel:
    """
    Quantum kernel for computing similarity between quantum states
    """
    
    def __init__(self, feature_map, quantum_instance=None):
        """
        Initialize quantum kernel
        
        Args:
            feature_map: Quantum circuit for feature encoding
            quantum_instance: Backend for quantum computation
        """
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_aer import Aer
        
        self.feature_map = feature_map
        
        # Use simulator if no backend provided
        if quantum_instance is None:
            backend = Aer.get_backend('aer_simulator')
            self.quantum_instance = backend
        else:
            self.quantum_instance = quantum_instance
        
        # Create quantum kernel
        self.kernel = FidelityQuantumKernel(
            feature_map=feature_map
        )
        
        print(f"✓ Quantum Kernel created")
        print(f"  • Backend: {type(self.quantum_instance).__name__}")
        print(f"  • Feature map depth: {feature_map.depth()}")
    
    def compute_kernel_matrix(self, X_train, X_test=None):
        """
        Compute quantum kernel matrix
        
        Args:
            X_train: Training data
            X_test: Test data (if None, compute training kernel)
            
        Returns:
            Kernel matrix
        """
        print(f"\n⚛️  Computing quantum kernel matrix...")
        print(f"   Training samples: {X_train.shape[0]}")
        if X_test is not None:
            print(f"   Test samples: {X_test.shape[0]}")
        
        if X_test is None:
            # Compute training kernel
            kernel_matrix = self.kernel.evaluate(X_train)
        else:
            # Compute test kernel
            kernel_matrix = self.kernel.evaluate(X_train, X_test)
        
        print(f"✓ Kernel matrix computed: {kernel_matrix.shape}")
        
        return kernel_matrix


def demo_quantum_encoding():
    """
    Demonstrate quantum feature encoding
    """
    print("\n" + "="*70)
    print("QUANTUM ENCODING DEMO")
    print("="*70)
    
    # Create sample features (10 samples, 131 features)
    np.random.seed(42)
    sample_features = np.random.randn(10, 131)
    
    # Test different encoding types
    for encoding_type in ['angle', 'pauli', 'zz']:
        print(f"\n{'='*70}")
        print(f"Testing {encoding_type.upper()} Encoding")
        print(f"{'='*70}")
        
        encoder = QuantumFeatureEncoder(
            n_features=131,
            encoding_type=encoding_type,
            n_qubits=10
        )
        
        # Encode features
        reduced_features = encoder.encode_features(sample_features)
        print(f"\n✓ Features encoded: {sample_features.shape} → {reduced_features.shape}")
        
        # Create and visualize circuit
        feature_map = encoder.create_feature_map(reps=2)
        print(f"✓ Feature map created")
        print(f"  • Depth: {feature_map.depth()}")
        print(f"  • Gates: {len(feature_map.data)}")
        print(f"  • Parameters: {feature_map.num_parameters}")
    
    print("\n" + "="*70)
    print("✅ QUANTUM ENCODING DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    demo_quantum_encoding()
