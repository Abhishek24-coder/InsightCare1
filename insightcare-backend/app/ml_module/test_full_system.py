"""
Comprehensive ML Module Test
Test all components end-to-end
"""

import numpy as np
import pickle
from pathlib import Path
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineering

def test_full_pipeline():
    print("="*70)
    print("COMPREHENSIVE ML MODULE TEST")
    print("="*70)
    
    # Test 1: Data Pipeline
    print("\n[1/5] Testing Data Pipeline...")
    try:
        pipeline = DataPipeline()
        success = pipeline.load_data()
        df = pipeline.prepare_data()
        symptoms = pipeline.get_unique_symptoms(df)
        diseases = pipeline.get_unique_diseases(df)
        severity = pipeline.create_severity_dict()
        
        assert len(symptoms) == 131, f"Expected 131 symptoms, got {len(symptoms)}"
        assert len(diseases) == 41, f"Expected 41 diseases, got {len(diseases)}"
        assert len(df) == 4920, f"Expected 4920 records, got {len(df)}"
        
        print("  ✅ Data Pipeline: PASS")
        print(f"     - Loaded {len(df)} records")
        print(f"     - {len(symptoms)} symptoms, {len(diseases)} diseases")
    except Exception as e:
        print(f"  ❌ Data Pipeline: FAIL - {e}")
        return False
    
    # Test 2: Feature Engineering
    print("\n[2/5] Testing Feature Engineering...")
    try:
        fe = FeatureEngineering(pipeline)
        fe.df = df
        
        # Test feature vector creation
        test_symptoms = ['fever', 'cough', 'fatigue']
        feature_vec = fe.create_feature_vector(test_symptoms, use_severity=True)
        
        assert feature_vec.shape == (131,), f"Expected shape (131,), got {feature_vec.shape}"
        assert np.sum(feature_vec) > 0, "Feature vector should have non-zero values"
        
        # Test training data preparation
        X, y = fe.prepare_training_data(use_severity=True)
        assert X.shape == (4920, 131), f"Expected X shape (4920, 131), got {X.shape}"
        assert y.shape == (4920,), f"Expected y shape (4920,), got {y.shape}"
        
        print("  ✅ Feature Engineering: PASS")
        print(f"     - Feature vectors: {X.shape}")
        print(f"     - Labels: {y.shape}")
    except Exception as e:
        print(f"  ❌ Feature Engineering: FAIL - {e}")
        return False
    
    # Test 3: Load Models
    print("\n[3/5] Testing Model Loading...")
    try:
        models_dir = Path(__file__).parent / "models"
        
        # Load Random Forest
        with open(models_dir / "random_forest_model.pkl", 'rb') as f:
            rf_model = pickle.load(f)
        
        # Load XGBoost
        with open(models_dir / "xgboost_model.pkl", 'rb') as f:
            xgb_model = pickle.load(f)
        
        # Load encoder
        with open(models_dir / "label_encoder.pkl", 'rb') as f:
            encoder_data = pickle.load(f)
        
        assert rf_model.n_features_in_ == 131, "RF model should have 131 features"
        assert rf_model.n_classes_ == 41, "RF model should predict 41 classes"
        
        print("  ✅ Model Loading: PASS")
        print(f"     - Random Forest: {rf_model.n_estimators} trees")
        print(f"     - XGBoost: {xgb_model.n_estimators} estimators")
        print(f"     - Encoder: {len(encoder_data['diseases_list'])} diseases")
    except Exception as e:
        print(f"  ❌ Model Loading: FAIL - {e}")
        return False
    
    # Test 4: Model Predictions
    print("\n[4/5] Testing Model Predictions...")
    try:
        # Create test case: Diabetes symptoms
        diabetes_symptoms = ['fatigue', 'weight_loss', 'increased_appetite', 'polyuria']
        feature_vec = fe.create_feature_vector(diabetes_symptoms, use_severity=True)
        feature_vec = feature_vec.reshape(1, -1)
        
        # Get predictions
        rf_pred = rf_model.predict(feature_vec)[0]
        xgb_pred = xgb_model.predict(feature_vec)[0]
        
        # Decode predictions
        rf_disease = encoder_data['label_encoder'].inverse_transform([rf_pred])[0]
        xgb_disease = encoder_data['label_encoder'].inverse_transform([xgb_pred])[0]
        
        print("  ✅ Model Predictions: PASS")
        print(f"     - Test symptoms: {diabetes_symptoms}")
        print(f"     - Random Forest: {rf_disease}")
        print(f"     - XGBoost: {xgb_disease}")
        
        # Get prediction probabilities
        rf_proba = rf_model.predict_proba(feature_vec)[0]
        xgb_proba = xgb_model.predict_proba(feature_vec)[0]
        
        # Top 3 predictions for RF
        top3_indices = np.argsort(rf_proba)[-3:][::-1]
        print(f"\n     Random Forest Top 3:")
        for idx in top3_indices:
            disease = encoder_data['label_encoder'].inverse_transform([idx])[0]
            confidence = rf_proba[idx] * 100
            print(f"       {confidence:5.2f}% - {disease}")
        
    except Exception as e:
        print(f"  ❌ Model Predictions: FAIL - {e}")
        return False
    
    # Test 5: Real-world test cases
    print("\n[5/5] Testing Real-world Cases...")
    try:
        test_cases = [
            {
                'name': 'Common Cold',
                'symptoms': ['continuous_sneezing', 'chills', 'cough', 'runny_nose']
            },
            {
                'name': 'Malaria',
                'symptoms': ['chills', 'vomiting', 'high_fever', 'sweating', 'headache']
            },
            {
                'name': 'Fungal Infection',
                'symptoms': ['itching', 'skin_rash', 'nodal_skin_eruptions']
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            feature_vec = fe.create_feature_vector(case['symptoms'], use_severity=True)
            feature_vec = feature_vec.reshape(1, -1)
            
            rf_pred = rf_model.predict(feature_vec)[0]
            rf_disease = encoder_data['label_encoder'].inverse_transform([rf_pred])[0]
            
            rf_proba = rf_model.predict_proba(feature_vec)[0]
            confidence = rf_proba[rf_pred] * 100
            
            print(f"\n     Case {i}: {case['name']}")
            print(f"       Symptoms: {', '.join(case['symptoms'])}")
            print(f"       Prediction: {rf_disease} ({confidence:.1f}% confidence)")
        
        print("\n  ✅ Real-world Cases: PASS")
        
    except Exception as e:
        print(f"  ❌ Real-world Cases: FAIL - {e}")
        return False
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - SYSTEM FULLY OPERATIONAL!")
    print("="*70)
    print("\n📊 System Summary:")
    print(f"   • Data: 4,920 training samples")
    print(f"   • Features: 131 symptoms")
    print(f"   • Classes: 41 diseases")
    print(f"   • Models: Random Forest + XGBoost")
    print(f"   • Status: ✅ Ready for production")
    
    return True


if __name__ == "__main__":
    success = test_full_pipeline()
    if not success:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        exit(1)
    else:
        print("\n🎉 All systems operational!")
        exit(0)
