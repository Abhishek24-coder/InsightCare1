"""
Model Evaluation - Detailed accuracy and performance metrics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from pathlib import Path

from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineering


def evaluate_saved_models():
    """Evaluate the accuracy of saved models"""
    
    print("="*70)
    print("DETAILED MODEL ACCURACY EVALUATION")
    print("="*70)
    
    # Load data
    print("\n[1] Loading data...")
    pipeline = DataPipeline()
    pipeline.load_data()
    df = pipeline.prepare_data()
    pipeline.get_unique_symptoms(df)
    pipeline.get_unique_diseases(df)
    pipeline.create_severity_dict()
    
    # Prepare features
    print("\n[2] Preparing features...")
    fe = FeatureEngineering(pipeline)
    fe.df = df
    X, y = fe.prepare_training_data(use_severity=True)
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Load models
    print("\n[3] Loading trained models...")
    models_dir = Path(__file__).parent / "models"
    
    with open(models_dir / "random_forest_model.pkl", 'rb') as f:
        rf_model = pickle.load(f)
    
    with open(models_dir / "xgboost_model.pkl", 'rb') as f:
        xgb_model = pickle.load(f)
    
    with open(models_dir / "label_encoder.pkl", 'rb') as f:
        encoder_data = pickle.load(f)
        label_encoder = encoder_data['label_encoder']
    
    # Evaluate Random Forest
    print("\n" + "="*70)
    print("RANDOM FOREST ACCURACY")
    print("="*70)
    
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    
    rf_train_acc = accuracy_score(y_train, rf_train_pred)
    rf_test_acc = accuracy_score(y_test, rf_test_pred)
    
    print(f"\n📊 Training Accuracy: {rf_train_acc:.4f} ({rf_train_acc*100:.2f}%)")
    print(f"📊 Test Accuracy:     {rf_test_acc:.4f} ({rf_test_acc*100:.2f}%)")
    print(f"📊 Accuracy Drop:     {(rf_train_acc - rf_test_acc)*100:.2f}%")
    
    if rf_train_acc - rf_test_acc > 0.05:
        print("⚠️  Warning: Model may be overfitting (>5% accuracy drop)")
    else:
        print("✅ Model is not overfitting")
    
    # Evaluate XGBoost
    print("\n" + "="*70)
    print("XGBOOST ACCURACY")
    print("="*70)
    
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)
    
    xgb_train_acc = accuracy_score(y_train, xgb_train_pred)
    xgb_test_acc = accuracy_score(y_test, xgb_test_pred)
    
    print(f"\n📊 Training Accuracy: {xgb_train_acc:.4f} ({xgb_train_acc*100:.2f}%)")
    print(f"📊 Test Accuracy:     {xgb_test_acc:.4f} ({xgb_test_acc*100:.2f}%)")
    print(f"📊 Accuracy Drop:     {(xgb_train_acc - xgb_test_acc)*100:.2f}%")
    
    if xgb_train_acc - xgb_test_acc > 0.05:
        print("⚠️  Warning: Model may be overfitting (>5% accuracy drop)")
    else:
        print("✅ Model is not overfitting")
    
    # Compare models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Random Forest':<20} {'XGBoost':<20}")
    print("-"*70)
    print(f"{'Training Accuracy':<25} {rf_train_acc*100:>6.2f}%            {xgb_train_acc*100:>6.2f}%")
    print(f"{'Test Accuracy':<25} {rf_test_acc*100:>6.2f}%            {xgb_test_acc*100:>6.2f}%")
    
    winner = "Random Forest" if rf_test_acc > xgb_test_acc else "XGBoost" if xgb_test_acc > rf_test_acc else "Tie"
    print(f"\n🏆 Better Model: {winner}")
    
    # Target achievement
    print("\n" + "="*70)
    print("TARGET ACHIEVEMENT")
    print("="*70)
    
    target = 0.75  # 75% target
    
    print(f"\n🎯 Target Accuracy: {target*100:.0f}%")
    print(f"\n📊 Random Forest: {rf_test_acc*100:.2f}%")
    if rf_test_acc >= target:
        print(f"   ✅ ACHIEVED! ({(rf_test_acc-target)*100:.2f}% above target)")
    else:
        print(f"   ❌ Below target by {(target-rf_test_acc)*100:.2f}%")
    
    print(f"\n📊 XGBoost: {xgb_test_acc*100:.2f}%")
    if xgb_test_acc >= target:
        print(f"   ✅ ACHIEVED! ({(xgb_test_acc-target)*100:.2f}% above target)")
    else:
        print(f"   ❌ Below target by {(target-xgb_test_acc)*100:.2f}%")
    
    # Per-class accuracy
    print("\n" + "="*70)
    print("PER-DISEASE ACCURACY (Random Forest - Sample)")
    print("="*70)
    
    diseases = label_encoder.inverse_transform(np.unique(y_test))
    correct_per_disease = {}
    total_per_disease = {}
    
    for i, (true, pred) in enumerate(zip(y_test, rf_test_pred)):
        disease = label_encoder.inverse_transform([true])[0]
        if disease not in total_per_disease:
            total_per_disease[disease] = 0
            correct_per_disease[disease] = 0
        total_per_disease[disease] += 1
        if true == pred:
            correct_per_disease[disease] += 1
    
    # Show top 10 and bottom 10
    accuracies = []
    for disease in total_per_disease:
        acc = correct_per_disease[disease] / total_per_disease[disease]
        accuracies.append((disease, acc, total_per_disease[disease]))
    
    accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print("\n✅ Top 10 Best Predicted Diseases:")
    for i, (disease, acc, count) in enumerate(accuracies[:10], 1):
        print(f"   {i:2d}. {disease:40s} {acc*100:5.1f}% ({count} samples)")
    
    print("\n⚠️  Bottom 10 Worst Predicted Diseases:")
    for i, (disease, acc, count) in enumerate(accuracies[-10:], 1):
        print(f"   {i:2d}. {disease:40s} {acc*100:5.1f}% ({count} samples)")
    
    # Overall assessment
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)
    
    avg_acc = (rf_test_acc + xgb_test_acc) / 2
    
    print(f"\n📊 Average Test Accuracy: {avg_acc*100:.2f}%")
    
    if avg_acc >= 0.95:
        print("🌟 EXCELLENT - Models are performing exceptionally well!")
    elif avg_acc >= 0.85:
        print("✅ VERY GOOD - Models are performing above expectations!")
    elif avg_acc >= 0.75:
        print("✅ GOOD - Models meet the target requirements!")
    elif avg_acc >= 0.65:
        print("⚠️  FAIR - Models need some improvement")
    else:
        print("❌ POOR - Models need significant improvement")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if rf_train_acc - rf_test_acc > 0.05 or xgb_train_acc - xgb_test_acc > 0.05:
        print("\n⚠️  OVERFITTING DETECTED:")
        print("   • Reduce model complexity")
        print("   • Increase regularization")
        print("   • Get more training data")
        print("   • Use cross-validation")
    
    if avg_acc < 0.75:
        print("\n📈 TO IMPROVE ACCURACY:")
        print("   • Hyperparameter tuning (GridSearch)")
        print("   • Try ensemble methods (voting classifier)")
        print("   • Feature engineering (add more features)")
        print("   • Balance dataset (some diseases have few samples)")
    else:
        print("\n✅ Models are performing well!")
        print("   • Current accuracy is excellent for medical diagnosis")
        print("   • No immediate improvements needed")
        print("   • Can be deployed to production")
    
    return {
        'rf_test_acc': rf_test_acc,
        'xgb_test_acc': xgb_test_acc,
        'avg_acc': avg_acc
    }


if __name__ == "__main__":
    results = evaluate_saved_models()
