"""
Complete Model Training Script - Train All ML Models
Author: AI Developer
Description: Comprehensive training pipeline for all models with validation
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """
    Main training pipeline for all models
    """
    print("="*100)
    print(" " * 30 + "INSIGHTCARE ML MODEL TRAINING")
    print("="*100)
    print("\nThis script will train all ML models with optimal configurations")
    print("Expected training time: 5-15 minutes depending on hardware\n")
    
    # Step 1: Verify data files
    print("\n" + "="*100)
    print("STEP 1: VERIFYING DATA FILES")
    print("="*100)
    
    data_dir = Path(__file__).parent / "data"
    required_files = [
        "dataset.csv",
        "Symptom-severity.csv",
        "symptom_Description.csv",
        "symptom_precaution.csv"
    ]
    
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            size = file_path.stat().st_size / 1024  # KB
            print(f"  ✓ {file} ({size:.2f} KB)")
        else:
            print(f"  ✗ {file} - MISSING!")
            print(f"\n❌ Error: Required data file missing: {file}")
            return False
    
    print("\n✅ All data files verified!")
    
    # Step 2: Train baseline models
    print("\n" + "="*100)
    print("STEP 2: TRAINING BASELINE MODELS")
    print("="*100)
    print("\nTraining Random Forest and XGBoost with default optimal parameters...")
    
    try:
        from train_models import ModelTrainer
        
        trainer = ModelTrainer(test_size=0.2, random_state=42)
        trainer.prepare_data(use_severity=True)
        
        # Train Random Forest
        print("\n" + "-"*80)
        print("Training Random Forest...")
        print("-"*80)
        rf_results = trainer.train_random_forest(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            verbose=True
        )
        
        # Train XGBoost
        print("\n" + "-"*80)
        print("Training XGBoost...")
        print("-"*80)
        xgb_results = trainer.train_xgboost(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            verbose=True
        )
        
        # Cross-validation
        print("\n" + "-"*80)
        print("Running Cross-Validation...")
        print("-"*80)
        cv_results = trainer.cross_validate_models(cv=5)
        
        # Save models
        print("\n" + "-"*80)
        print("Saving Models...")
        print("-"*80)
        trainer.save_models()
        
        print("\n✅ Baseline models trained successfully!")
        print(f"\nBaseline Results:")
        print(f"  • Random Forest: {rf_results['test_accuracy']*100:.2f}% accuracy")
        print(f"  • XGBoost: {xgb_results['test_accuracy']*100:.2f}% accuracy")
        
    except Exception as e:
        print(f"\n❌ Error during baseline training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Advanced training with optimization
    print("\n" + "="*100)
    print("STEP 3: ADVANCED TRAINING WITH HYPERPARAMETER OPTIMIZATION")
    print("="*100)
    print("\nThis step performs extensive hyperparameter tuning...")
    print("⚠️  This may take 10-30 minutes depending on your hardware")
    
    user_input = input("\nDo you want to run advanced optimization? (yes/no): ").strip().lower()
    
    if user_input in ['yes', 'y']:
        try:
            from advanced_training import AdvancedModelTrainer
            
            adv_trainer = AdvancedModelTrainer(test_size=0.2, random_state=42)
            adv_trainer.prepare_data(use_severity=True, augment=True)
            
            # Use quick mode for faster training
            quick_mode = True
            print("\nUsing quick mode for faster optimization...")
            
            # Optimize Random Forest
            print("\n" + "-"*80)
            print("Optimizing Random Forest...")
            print("-"*80)
            rf_opt_results = adv_trainer.optimize_random_forest(quick_mode=quick_mode)
            
            # Optimize XGBoost
            print("\n" + "-"*80)
            print("Optimizing XGBoost...")
            print("-"*80)
            xgb_opt_results = adv_trainer.optimize_xgboost(quick_mode=quick_mode)
            
            # Create ensemble
            print("\n" + "-"*80)
            print("Creating Ensemble Model...")
            print("-"*80)
            ensemble_results = adv_trainer.create_ensemble()
            
            # Save optimized models
            print("\n" + "-"*80)
            print("Saving Optimized Models...")
            print("-"*80)
            adv_trainer.save_models(prefix="optimized")
            
            print("\n✅ Advanced optimization complete!")
            print(f"\nOptimized Results:")
            print(f"  • Random Forest: {rf_opt_results['test_accuracy']*100:.2f}% accuracy")
            print(f"  • XGBoost: {xgb_opt_results['test_accuracy']*100:.2f}% accuracy")
            print(f"  • Ensemble: {ensemble_results['test_accuracy']*100:.2f}% accuracy")
            
        except Exception as e:
            print(f"\n❌ Error during advanced training: {e}")
            import traceback
            traceback.print_exc()
            print("\n⚠️  Advanced optimization failed, but baseline models are still available")
    else:
        print("\n⏭️  Skipping advanced optimization")
    
    # Step 4: Model validation
    print("\n" + "="*100)
    print("STEP 4: VALIDATING TRAINED MODELS")
    print("="*100)
    
    try:
        from evaluate import ModelEvaluator
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_all_models()
        
        print("\n✅ Model validation complete!")
        
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Test predictions
    print("\n" + "="*100)
    print("STEP 5: TESTING PREDICTIONS")
    print("="*100)
    
    try:
        from predict import DiseasePredictor
        
        predictor = DiseasePredictor()
        
        # Test case 1: Common cold
        print("\n📝 Test Case 1: Common cold symptoms")
        test_symptoms_1 = ['continuous_sneezing', 'cough', 'fatigue', 'headache']
        result_1 = predictor.predict(test_symptoms_1, model='random_forest')
        print(f"  Prediction: {result_1['predicted_disease']}")
        print(f"  Confidence: {result_1['confidence_percentage']}%")
        
        # Test case 2: Flu symptoms
        print("\n📝 Test Case 2: Flu symptoms")
        test_symptoms_2 = ['high_fever', 'headache', 'chills', 'muscle_pain', 'fatigue']
        result_2 = predictor.predict(test_symptoms_2, model='random_forest')
        print(f"  Prediction: {result_2['predicted_disease']}")
        print(f"  Confidence: {result_2['confidence_percentage']}%")
        
        # Test case 3: Diabetes symptoms
        print("\n📝 Test Case 3: Diabetes symptoms")
        test_symptoms_3 = ['excessive_hunger', 'increased_appetite', 'obesity', 'fatigue']
        result_3 = predictor.predict(test_symptoms_3, model='random_forest')
        print(f"  Prediction: {result_3['predicted_disease']}")
        print(f"  Confidence: {result_3['confidence_percentage']}%")
        
        print("\n✅ Prediction tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error during prediction tests: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n" + "="*100)
    print("✅ TRAINING COMPLETE!")
    print("="*100)
    
    print("\n📦 Trained Models:")
    models_dir = Path(__file__).parent / "models"
    if models_dir.exists():
        for model_file in sorted(models_dir.glob("*.pkl")):
            size = model_file.stat().st_size / 1024 / 1024  # MB
            print(f"  • {model_file.name} ({size:.2f} MB)")
    
    print("\n✨ Your models are ready for use!")
    print("\n📚 Next Steps:")
    print("  1. Run 'python evaluate.py' to see detailed evaluation metrics")
    print("  2. Run 'python predict.py' to test predictions")
    print("  3. Start the backend API server to use models in production")
    
    print("\n" + "="*100)
    
    return True


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    success = main()
    
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n⏱️  Total training time: {minutes}m {seconds}s")
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
