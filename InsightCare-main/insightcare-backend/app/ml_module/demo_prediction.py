"""
Simple example of using the Prediction API
"""

from predict import DiseasePredictor

# Initialize predictor (loads all models)
predictor = DiseasePredictor()

print("="*70)
print("DISEASE PREDICTION DEMO")
print("="*70)

# Example 1: Predict Diabetes
print("\n📋 Example 1: Patient with diabetes symptoms")
print("Symptoms: fatigue, weight_loss, polyuria")

result = predictor.predict(
    symptoms=['fatigue', 'weight_loss', 'polyuria'],
    model='random_forest',
    top_k=3
)

print(f"\n🏥 Prediction: {result['predicted_disease']}")
print(f"📊 Confidence: {result['confidence_percentage']}%")
print(f"\nTop 3 Possible Diseases:")
for i, pred in enumerate(result['top_predictions'], 1):
    print(f"  {i}. {pred['disease']:30s} - {pred['confidence_percentage']}%")

# Example 2: Predict Malaria
print("\n" + "="*70)
print("\n📋 Example 2: Patient with malaria symptoms")
print("Symptoms: high_fever, chills, sweating, headache")

result2 = predictor.predict(
    symptoms=['high_fever', 'chills', 'sweating', 'headache'],
    model='random_forest',
    top_k=3
)

print(f"\n🏥 Prediction: {result2['predicted_disease']}")
print(f"📊 Confidence: {result2['confidence_percentage']}%")
print(f"\nTop 3 Possible Diseases:")
for i, pred in enumerate(result2['top_predictions'], 1):
    print(f"  {i}. {pred['disease']:30s} - {pred['confidence_percentage']}%")

# Example 3: Compare both models
print("\n" + "="*70)
print("\n📋 Example 3: Compare Random Forest vs XGBoost")
print("Symptoms: itching, skin_rash")

both_results = predictor.predict_with_both_models(
    symptoms=['itching', 'skin_rash'],
    top_k=3
)

print(f"\n🌲 Random Forest says: {both_results['random_forest']['predicted_disease']}")
print(f"   Confidence: {both_results['random_forest']['confidence_percentage']}%")

print(f"\n🚀 XGBoost says: {both_results['xgboost']['predicted_disease']}")
print(f"   Confidence: {both_results['xgboost']['confidence_percentage']}%")

print(f"\n{'✅ Models AGREE' if both_results['models_agree'] else '⚠️  Models DISAGREE'}")

print("\n" + "="*70)
print("✅ DEMO COMPLETE")
print("="*70)
