"""
Feature Engineering - Convert Symptoms to ML Features
Author: Abhishek
Description: Transform symptom data into numerical feature vectors for ML models
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import LabelEncoder
from data_pipeline import DataPipeline
import pickle


class FeatureEngineering:
    """
    Converts symptom data into numerical features for machine learning
    """
    
    def __init__(self, pipeline: DataPipeline = None):
        """
        Initialize feature engineering
        
        Args:
            pipeline: DataPipeline instance (will create new one if None)
        """
        if pipeline is None:
            self.pipeline = DataPipeline()
            self.pipeline.load_data()
            self.df = self.pipeline.prepare_data()
            self.symptoms_list = self.pipeline.get_unique_symptoms(self.df)
            self.diseases_list = self.pipeline.get_unique_diseases(self.df)
            self.severity_dict = self.pipeline.create_severity_dict()
        else:
            self.pipeline = pipeline
            self.df = pipeline.df_disease_symptoms
            self.symptoms_list = pipeline.symptoms_list
            self.diseases_list = pipeline.diseases_list
            self.severity_dict = pipeline.severity_dict
        
        # Label encoder for diseases
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.diseases_list)
        
        print(f"\n✓ Feature Engineering initialized")
        print(f"  • Features (symptoms): {len(self.symptoms_list)}")
        print(f"  • Classes (diseases): {len(self.diseases_list)}")
    
    def create_feature_vector(self, symptoms: List[str], use_severity: bool = True) -> np.ndarray:
        """
        Create a feature vector from a list of symptoms
        
        Args:
            symptoms: List of symptom names
            use_severity: Whether to use severity weights (True) or binary encoding (False)
            
        Returns:
            Feature vector (numpy array)
        """
        # Initialize feature vector with zeros
        feature_vector = np.zeros(len(self.symptoms_list))
        
        # Set values for present symptoms
        for symptom in symptoms:
            symptom = str(symptom).strip().lower().replace(' ', '_')
            
            if symptom in self.symptoms_list:
                symptom_idx = self.symptoms_list.index(symptom)
                
                if use_severity and symptom in self.severity_dict:
                    # Use severity weight
                    feature_vector[symptom_idx] = self.severity_dict[symptom]
                else:
                    # Binary encoding (0 or 1)
                    feature_vector[symptom_idx] = 1
        
        return feature_vector
    
    def prepare_training_data(self, use_severity: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare X (features) and y (labels) for training
        
        Args:
            use_severity: Whether to use severity weights
            
        Returns:
            Tuple of (X, y) where X is feature matrix and y is label vector
        """
        if self.df is None:
            raise ValueError("Data not loaded. Initialize with pipeline or load data.")
        
        print(f"\n{'='*60}")
        print("PREPARING TRAINING DATA")
        print(f"{'='*60}")
        
        # Get symptom columns
        symptom_cols = [col for col in self.df.columns if col.startswith('Symptom_')]
        
        X_list = []
        y_list = []
        
        # Process each record
        for idx, row in self.df.iterrows():
            # Get symptoms for this record
            symptoms = []
            for col in symptom_cols:
                if pd.notna(row[col]) and row[col]:
                    symptoms.append(row[col])
            
            # Create feature vector
            feature_vector = self.create_feature_vector(symptoms, use_severity)
            X_list.append(feature_vector)
            
            # Get disease label
            disease = row['Disease']
            y_list.append(disease)
        
        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        print(f"\n✓ Training data prepared:")
        print(f"  • X shape: {X.shape} (samples × features)")
        print(f"  • y shape: {y_encoded.shape} (samples)")
        print(f"  • Feature encoding: {'Severity-weighted' if use_severity else 'Binary'}")
        print(f"  • Non-zero features: {np.count_nonzero(X)} / {X.size}")
        print(f"  • Sparsity: {(1 - np.count_nonzero(X) / X.size) * 100:.2f}%")
        
        return X, y_encoded
    
    def encode_disease(self, disease: str) -> int:
        """
        Encode disease name to numerical label
        
        Args:
            disease: Disease name
            
        Returns:
            Encoded label
        """
        return self.label_encoder.transform([disease])[0]
    
    def decode_disease(self, label: int) -> str:
        """
        Decode numerical label to disease name
        
        Args:
            label: Encoded label
            
        Returns:
            Disease name
        """
        return self.label_encoder.inverse_transform([label])[0]
    
    def decode_diseases(self, labels: np.ndarray) -> List[str]:
        """
        Decode multiple numerical labels to disease names
        
        Args:
            labels: Array of encoded labels
            
        Returns:
            List of disease names
        """
        return self.label_encoder.inverse_transform(labels).tolist()
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names (symptoms)
        
        Returns:
            List of symptom names
        """
        return self.symptoms_list
    
    def get_class_names(self) -> List[str]:
        """
        Get the list of class names (diseases)
        
        Returns:
            List of disease names
        """
        return self.diseases_list
    
    def save_encoders(self, filepath: str = None):
        """
        Save label encoder and feature information
        
        Args:
            filepath: Path to save the encoder
        """
        if filepath is None:
            from pathlib import Path
            filepath = Path(__file__).parent / "models" / "label_encoder.pkl"
        
        encoder_data = {
            'label_encoder': self.label_encoder,
            'symptoms_list': self.symptoms_list,
            'diseases_list': self.diseases_list,
            'severity_dict': self.severity_dict
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(encoder_data, f)
        
        print(f"\n✓ Encoders saved to: {filepath}")
    
    def load_encoders(self, filepath: str = None):
        """
        Load label encoder and feature information
        
        Args:
            filepath: Path to load the encoder from
        """
        if filepath is None:
            from pathlib import Path
            filepath = Path(__file__).parent / "models" / "label_encoder.pkl"
        
        with open(filepath, 'rb') as f:
            encoder_data = pickle.load(f)
        
        self.label_encoder = encoder_data['label_encoder']
        self.symptoms_list = encoder_data['symptoms_list']
        self.diseases_list = encoder_data['diseases_list']
        self.severity_dict = encoder_data['severity_dict']
        
        print(f"\n✓ Encoders loaded from: {filepath}")
    
    def visualize_sample_features(self, n_samples: int = 3):
        """
        Visualize sample feature vectors
        
        Args:
            n_samples: Number of samples to visualize
        """
        print(f"\n{'='*70}")
        print("SAMPLE FEATURE VECTORS")
        print(f"{'='*70}")
        
        if self.df is None:
            print("No data loaded!")
            return
        
        symptom_cols = [col for col in self.df.columns if col.startswith('Symptom_')]
        
        for i in range(min(n_samples, len(self.df))):
            row = self.df.iloc[i]
            disease = row['Disease']
            
            # Get symptoms
            symptoms = []
            for col in symptom_cols:
                if pd.notna(row[col]) and row[col]:
                    symptoms.append(row[col])
            
            # Create feature vector
            feature_vector = self.create_feature_vector(symptoms, use_severity=True)
            
            # Show non-zero features
            non_zero_indices = np.nonzero(feature_vector)[0]
            
            print(f"\nSample {i+1}:")
            print(f"  Disease: {disease}")
            print(f"  Symptoms ({len(symptoms)}):")
            for symptom in symptoms:
                severity = self.severity_dict.get(symptom, 'N/A')
                print(f"    • {symptom:40s} (weight: {severity})")
            
            print(f"  Feature Vector Shape: {feature_vector.shape}")
            print(f"  Non-zero features: {len(non_zero_indices)} / {len(feature_vector)}")
            print(f"  Encoded label: {self.encode_disease(disease)}")


def main():
    """Test feature engineering"""
    print("Testing Feature Engineering...\n")
    
    # Initialize pipeline
    pipeline = DataPipeline()
    pipeline.load_data()
    df = pipeline.prepare_data()
    symptoms = pipeline.get_unique_symptoms(df)
    diseases = pipeline.get_unique_diseases(df)
    severity_dict = pipeline.create_severity_dict()
    
    # Initialize feature engineering
    fe = FeatureEngineering(pipeline)
    fe.df = df  # Set the cleaned dataframe
    
    # Prepare training data
    X, y = fe.prepare_training_data(use_severity=True)
    
    # Visualize sample features
    fe.visualize_sample_features(n_samples=3)
    
    # Test encoding/decoding
    print(f"\n{'='*70}")
    print("ENCODING/DECODING TEST")
    print(f"{'='*70}")
    test_disease = diseases[0]
    encoded = fe.encode_disease(test_disease)
    decoded = fe.decode_disease(encoded)
    print(f"\nOriginal: {test_disease}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    print(f"Match: {test_disease == decoded}")
    
    # Save encoders
    fe.save_encoders()
    
    print(f"\n{'='*70}")
    print("✅ Feature Engineering Complete!")
    print(f"{'='*70}")
    print(f"\n📊 Summary:")
    print(f"  • Training samples: {X.shape[0]}")
    print(f"  • Features: {X.shape[1]}")
    print(f"  • Classes: {len(diseases)}")
    print(f"  • Ready for model training!")


if __name__ == "__main__":
    main()
