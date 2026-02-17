"""
Quick Model Verification
Check if trained models are working
"""

import pickle
from pathlib import Path
import numpy as np

def check_models():
    print("="*70)
    print("CHECKING SAVED MODELS")
    print("="*70)
    
    models_dir = Path(__file__).parent / "models"
    
    # Check files
    files = {
        'Random Forest': models_dir / "random_forest_model.pkl",
        'XGBoost': models_dir / "xgboost_model.pkl",
        'Label Encoder': models_dir / "label_encoder.pkl"
    }
    
    print("\n📁 Model Files:")
    for name, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name:20s}: {size_mb:.2f} MB")
        else:
            print(f"  ✗ {name:20s}: NOT FOUND")
    
    # Load and test models
    print("\n🔄 Loading models...")
    
    try:
        # Load Random Forest
        with open(files['Random Forest'], 'rb') as f:
            rf_model = pickle.load(f)
        print(f"  ✓ Random Forest loaded")
        print(f"    - n_estimators: {rf_model.n_estimators}")
        print(f"    - n_features: {rf_model.n_features_in_}")
        print(f"    - n_classes: {rf_model.n_classes_}")
        
        # Load XGBoost
        with open(files['XGBoost'], 'rb') as f:
            xgb_model = pickle.load(f)
        print(f"  ✓ XGBoost loaded")
        print(f"    - n_estimators: {xgb_model.n_estimators}")
        
        # Load encoder
        with open(files['Label Encoder'], 'rb') as f:
            encoder_data = pickle.load(f)
        print(f"  ✓ Label Encoder loaded")
        print(f"    - Symptoms: {len(encoder_data['symptoms_list'])}")
        print(f"    - Diseases: {len(encoder_data['diseases_list'])}")
        
        # Test prediction
        print("\n🧪 Testing prediction with dummy input...")
        dummy_input = np.random.randint(0, 5, size=(1, rf_model.n_features_in_))
        
        rf_pred = rf_model.predict(dummy_input)
        xgb_pred = xgb_model.predict(dummy_input)
        
        rf_disease = encoder_data['label_encoder'].inverse_transform(rf_pred)[0]
        xgb_disease = encoder_data['label_encoder'].inverse_transform(xgb_pred)[0]
        
        print(f"  • Random Forest prediction: {rf_disease}")
        print(f"  • XGBoost prediction: {xgb_disease}")
        
        print("\n" + "="*70)
        print("✅ ALL MODELS ARE WORKING!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    check_models()
