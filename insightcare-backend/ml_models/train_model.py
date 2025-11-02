"""
Complete ML Training Pipeline - Random Forest & XGBoost
Train disease diagnosis models and save for production use
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ========== STEP 1: LOAD DATA ==========
print("=" * 80)
print("🚀 DISEASE DIAGNOSIS MODEL TRAINING")
print("=" * 80)

# Load dataset
data_path = 'data/dataset.csv'
if not os.path.exists(data_path):
    print(f"\n❌ ERROR: Dataset not found at {data_path}")
    print("Please download from: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset")
    print("And extract to: ml_models/data/")
    exit(1)

df = pd.read_csv(data_path)
print(f"\n✅ Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")

# ========== STEP 2: EXPLORE DATA ==========
print("\n" + "=" * 80)
print("📊 DATA EXPLORATION")
print("=" * 80)

print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3))

# Check diseases
disease_counts = df['Disease'].value_counts()
print(f"\n📈 Disease Statistics:")
print(f"   Total diseases: {len(disease_counts)}")
print(f"   Total patients: {len(df)}")
print(f"   Most common: {disease_counts.index[0]} ({disease_counts.iloc[0]} cases)")
print(f"   Least common: {disease_counts.index[-1]} ({disease_counts.iloc[-1]} cases)")

print(f"\n   Top 10 diseases:")
for i, (disease, count) in enumerate(disease_counts.head(10).items(), 1):
    print(f"   {i:2d}. {disease:30s} - {count:4d} cases")

# ========== STEP 3: CLEAN DATA ==========
print("\n" + "=" * 80)
print("🧹 DATA CLEANING")
print("=" * 80)

# Check missing values
missing = df.isnull().sum().sum()
print(f"\nMissing values: {missing}")

# Fill missing with empty string
df = df.fillna('')
print("✅ Missing values handled")

# ========== STEP 4: PREPARE FEATURES ==========
print("\n" + "=" * 80)
print("🔧 FEATURE ENGINEERING")
print("=" * 80)

# Separate features and target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Get symptom columns
symptom_cols = [col for col in X.columns if 'Symptom' in col]
print(f"\nSymptom columns: {len(symptom_cols)}")

# Create symptom lists (combine all symptom columns)
X_processed = []
for idx, row in X.iterrows():
    patient_symptoms = [symptom for symptom in row if symptom != '']
    X_processed.append(patient_symptoms)

# Get all unique symptoms
all_symptoms = set()
for symptoms in X_processed:
    all_symptoms.update(symptoms)

print(f"Total unique symptoms: {len(all_symptoms)}")

# One-Hot Encode symptoms
print("\n🔢 Encoding symptoms (One-Hot Encoding)...")
mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X_processed)

print(f"Encoded shape: {X_encoded.shape}")
print(f"   {X_encoded.shape[0]} patients × {X_encoded.shape[1]} symptoms")
print(f"   Each cell: 1 = patient has symptom, 0 = doesn't have")

# ========== STEP 5: TRAIN/TEST SPLIT ==========
print("\n" + "=" * 80)
print("✂️  TRAIN/TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_encoded)*100:.1f}%)")
print(f"Testing set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X_encoded)*100:.1f}%)")

# ========== STEP 6: TRAIN RANDOM FOREST ==========
print("\n" + "=" * 80)
print("🌲 TRAINING RANDOM FOREST CLASSIFIER")
print("=" * 80)

print("\nHyperparameters:")
print("   n_estimators: 100 (number of trees)")
print("   max_depth: 20 (maximum tree depth)")
print("   random_state: 42 (reproducible results)")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

print("\n⏳ Training... (this may take 2-5 minutes)")
start_time = time.time()
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - start_time

print(f"✅ Training complete in {rf_train_time:.2f} seconds")

# Evaluate Random Forest
print("\n📊 Evaluating Random Forest...")
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"\n🎯 RANDOM FOREST RESULTS:")
print(f"   Accuracy: {rf_accuracy*100:.2f}%")
print(f"   Training time: {rf_train_time:.2f}s")

if rf_accuracy >= 0.75:
    print(f"   ✅ TARGET ACHIEVED! (75%+ accuracy)")
else:
    print(f"   ⚠️  Below 75% target")

# Detailed classification report
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))

# ========== STEP 7: TRAIN XGBOOST ==========
print("\n" + "=" * 80)
print("⚡ TRAINING XGBOOST CLASSIFIER")
print("=" * 80)

print("\nHyperparameters:")
print("   n_estimators: 100 (boosting rounds)")
print("   max_depth: 6 (tree depth)")
print("   learning_rate: 0.1")

# Encode labels for XGBoost
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

print("\n⏳ Training... (this may take 3-7 minutes)")
start_time = time.time()
xgb_model.fit(X_train, y_train_encoded)
xgb_train_time = time.time() - start_time

print(f"✅ Training complete in {xgb_train_time:.2f} seconds")

# Evaluate XGBoost
print("\n📊 Evaluating XGBoost...")
y_pred_xgb = xgb_model.predict(X_test)
y_pred_xgb_labels = le.inverse_transform(y_pred_xgb)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb_labels)

print(f"\n🎯 XGBOOST RESULTS:")
print(f"   Accuracy: {xgb_accuracy*100:.2f}%")
print(f"   Training time: {xgb_train_time:.2f}s")

if xgb_accuracy >= 0.80:
    print(f"   ✅ EXCELLENT! (80%+ accuracy)")
elif xgb_accuracy >= 0.75:
    print(f"   ✅ TARGET ACHIEVED! (75%+ accuracy)")
else:
    print(f"   ⚠️  Below 75% target")

print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred_xgb_labels, zero_division=0))

# ========== STEP 8: COMPARE MODELS ==========
print("\n" + "=" * 80)
print("⚖️  MODEL COMPARISON")
print("=" * 80)

print(f"\n{'Model':<20} {'Accuracy':<12} {'Training Time':<15} {'Status'}")
print("-" * 60)
print(f"{'Random Forest':<20} {rf_accuracy*100:>6.2f}%      {rf_train_time:>6.2f}s        {'✅' if rf_accuracy >= 0.75 else '⚠️'}")
print(f"{'XGBoost':<20} {xgb_accuracy*100:>6.2f}%      {xgb_train_time:>6.2f}s        {'✅' if xgb_accuracy >= 0.75 else '⚠️'}")

# Select best model
best_model = rf_model if rf_accuracy >= xgb_accuracy else xgb_model
best_model_name = "Random Forest" if rf_accuracy >= xgb_accuracy else "XGBoost"
best_accuracy = max(rf_accuracy, xgb_accuracy)

print(f"\n🏆 BEST MODEL: {best_model_name} ({best_accuracy*100:.2f}% accuracy)")

# ========== STEP 9: SAVE MODELS ==========
print("\n" + "=" * 80)
print("💾 SAVING MODELS")
print("=" * 80)

# Create models directory
os.makedirs('models', exist_ok=True)

# Save Random Forest
rf_path = 'models/random_forest_model.pkl'
joblib.dump(rf_model, rf_path)
rf_size = os.path.getsize(rf_path) / 1024 / 1024
print(f"\n✅ Random Forest saved: {rf_path} ({rf_size:.2f} MB)")

# Save XGBoost
xgb_path = 'models/xgboost_model.pkl'
joblib.dump(xgb_model, xgb_path)
joblib.dump(le, 'models/label_encoder.pkl')  # Save label encoder
xgb_size = os.path.getsize(xgb_path) / 1024 / 1024
print(f"✅ XGBoost saved: {xgb_path} ({xgb_size:.2f} MB)")

# Save best model
best_path = 'models/best_model.pkl'
if best_model_name == "XGBoost":
    joblib.dump(xgb_model, best_path)
else:
    joblib.dump(rf_model, best_path)
print(f"✅ Best model saved: {best_path}")

# Save symptom encoder
encoder_path = 'models/symptom_encoder.pkl'
joblib.dump(mlb, encoder_path)
print(f"✅ Symptom encoder saved: {encoder_path}")

# Save metadata
metadata = {
    'rf_accuracy': rf_accuracy,
    'xgb_accuracy': xgb_accuracy,
    'best_model': best_model_name,
    'best_accuracy': best_accuracy,
    'n_diseases': len(disease_counts),
    'n_symptoms': len(all_symptoms),
    'n_samples': len(df),
    'training_date': time.strftime('%Y-%m-%d %H:%M:%S')
}
joblib.dump(metadata, 'models/metadata.pkl')
print(f"✅ Metadata saved: models/metadata.pkl")

# ========== STEP 10: TEST PREDICTIONS ==========
print("\n" + "=" * 80)
print("🧪 TESTING PREDICTIONS")
print("=" * 80)

def predict_disease(symptoms_list, model, encoder, label_encoder=None):
    """Predict disease from symptoms"""
    symptoms_encoded = encoder.transform([symptoms_list])
    
    if label_encoder:  # XGBoost
        pred_encoded = model.predict(symptoms_encoded)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]
        probabilities = model.predict_proba(symptoms_encoded)[0]
        top_3_idx = probabilities.argsort()[-3:][::-1]
        top_3 = [(label_encoder.inverse_transform([i])[0], probabilities[i]) for i in top_3_idx]
    else:  # Random Forest
        prediction = model.predict(symptoms_encoded)[0]
        probabilities = model.predict_proba(symptoms_encoded)[0]
        top_3_idx = probabilities.argsort()[-3:][::-1]
        top_3 = [(model.classes_[i], probabilities[i]) for i in top_3_idx]
    
    return {
        'disease': prediction,
        'confidence': max(probabilities),
        'top_3': top_3
    }

# Test cases
test_cases = [
    ['itching', 'skin_rash', 'nodal_skin_eruptions'],
    ['continuous_sneezing', 'shivering', 'chills'],
    ['stomach_pain', 'acidity', 'vomiting'],
    ['fatigue', 'weight_loss', 'restlessness'],
    ['high_fever', 'headache', 'nausea']
]

for i, symptoms in enumerate(test_cases, 1):
    print(f"\nTest Case {i}:")
    print(f"   Symptoms: {symptoms}")
    
    if best_model_name == "XGBoost":
        result = predict_disease(symptoms, xgb_model, mlb, le)
    else:
        result = predict_disease(symptoms, rf_model, mlb)
    
    print(f"   Prediction: {result['disease']}")
    print(f"   Confidence: {result['confidence']*100:.1f}%")
    print(f"   Top 3 possibilities:")
    for j, (disease, prob) in enumerate(result['top_3'], 1):
        print(f"      {j}. {disease}: {prob*100:.1f}%")

# ========== SUMMARY ==========
print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)

print(f"""
📊 Summary:
   - Dataset: {len(df)} patients, {len(disease_counts)} diseases
   - Random Forest: {rf_accuracy*100:.2f}% accuracy
   - XGBoost: {xgb_accuracy*100:.2f}% accuracy
   - Best Model: {best_model_name} ({best_accuracy*100:.2f}%)
   
📁 Saved Files:
   - models/random_forest_model.pkl
   - models/xgboost_model.pkl
   - models/best_model.pkl
   - models/symptom_encoder.pkl
   - models/label_encoder.pkl (for XGBoost)
   - models/metadata.pkl
   
🚀 Next Steps:
   1. Test predictions: python test_model.py
   2. Integrate with FastAPI: Create /api/ml/diagnose endpoint
   3. Deploy to Railway
   
📝 Integration:
   from disease_classifier import DiseaseClassifier
   classifier = DiseaseClassifier()
   result = classifier.predict(['fever', 'cough'])
""")

print("\n" + "=" * 80)
