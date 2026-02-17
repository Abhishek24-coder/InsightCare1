"""
Analyze why predictions are failing and improve accuracy
"""

from predict import DiseasePredictor
from data_pipeline import DataPipeline
import pandas as pd

print("\n" + "="*70)
print("ANALYZING PREDICTION ISSUES")
print("="*70)

# Load data
pipeline = DataPipeline()
pipeline.load_data()
df = pipeline.prepare_data()

# Check what symptoms actually lead to these diseases
problem_diseases = ['Diabetes ', 'Dengue', 'Pneumonia']

for disease in problem_diseases:
    print(f"\n{'='*70}")
    print(f"DISEASE: {disease}")
    print(f"{'='*70}")
    
    # Filter rows for this disease
    disease_data = df[df['Disease'] == disease]
    
    if len(disease_data) == 0:
        print(f"❌ Disease not found in dataset!")
        continue
    
    print(f"✓ Found {len(disease_data)} samples")
    
    # Get all symptoms for this disease
    symptom_cols = [col for col in df.columns if col != 'Disease']
    all_symptoms = set()
    
    for idx, row in disease_data.head(5).iterrows():
        print(f"\nSample {idx + 1}:")
        symptoms = []
        for col in symptom_cols:
            if pd.notna(row[col]) and row[col] != '':
                symptom = row[col].strip().replace('_', ' ')
                if symptom:
                    symptoms.append(symptom)
                    all_symptoms.add(symptom)
        print(f"  Symptoms: {', '.join(symptoms[:7])}")
    
    print(f"\n📊 Most common symptoms for {disease}:")
    symptom_counts = {}
    for idx, row in disease_data.iterrows():
        for col in symptom_cols:
            if pd.notna(row[col]) and row[col] != '':
                symptom = row[col].strip().replace('_', ' ')
                if symptom:
                    symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
    
    # Sort by frequency
    sorted_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (symptom, count) in enumerate(sorted_symptoms[:10], 1):
        percentage = (count / len(disease_data)) * 100
        print(f"   {i:2}. {symptom:<35} {count:3} samples ({percentage:5.1f}%)")

print("\n" + "="*70)
print("SOLUTION: Use these exact symptoms for better predictions!")
print("="*70)
