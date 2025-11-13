"""
Generate Symptom Guide for Accurate Predictions
"""

from data_pipeline import DataPipeline
import pandas as pd

print("\n" + "="*80)
print("SYMPTOM GUIDE FOR ACCURATE PREDICTIONS")
print("="*80)

# Load data
pipeline = DataPipeline()
pipeline.load_data()
df = pipeline.prepare_data()

print(f"\n📊 Dataset contains {df['Disease'].nunique()} diseases and {len(df)} samples\n")

# Get top diseases
diseases = ['Diabetes ', 'Dengue', 'Pneumonia', 'Malaria', 'Tuberculosis', 
            'Common Cold', 'Typhoid', 'Hepatitis B', 'AIDS', 'Gastroenteritis']

symptom_cols = [col for col in df.columns if col != 'Disease']

for disease in diseases:
    disease_data = df[df['Disease'] == disease]
    
    if len(disease_data) == 0:
        continue
    
    print(f"{'='*80}")
    print(f"🏥 {disease.strip()}")
    print(f"{'='*80}")
    
    # Count symptoms
    symptom_counts = {}
    for idx, row in disease_data.iterrows():
        for col in symptom_cols:
            if pd.notna(row[col]) and row[col] != '':
                symptom = row[col].strip().replace(' ', '_')
                if symptom:
                    symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
    
    # Sort by frequency and show top symptoms
    sorted_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n✅ KEY SYMPTOMS (Use these for best accuracy):")
    key_symptoms = [s for s, c in sorted_symptoms if c == len(disease_data)][:7]
    for i, symptom in enumerate(key_symptoms, 1):
        print(f"   {i}. {symptom}")
    
    print(f"\n⚡ SUPPORTING SYMPTOMS (Optional but helpful):")
    supporting = [s for s, c in sorted_symptoms if c >= len(disease_data) * 0.8 and c < len(disease_data)][:5]
    for symptom in supporting:
        print(f"   • {symptom}")
    
    print(f"\n💡 Example usage:")
    example_symptoms = key_symptoms[:5] if len(key_symptoms) >= 5 else key_symptoms + supporting[:5-len(key_symptoms)]
    print(f"   symptoms = {example_symptoms}")
    print()

print("="*80)
print("📝 IMPORTANT NOTES:")
print("="*80)
print("1. Use underscores in symptom names (e.g., 'high_fever', not 'high fever')")
print("2. Include 5-7 symptoms for best results")
print("3. Start with KEY SYMPTOMS (100% frequency in dataset)")
print("4. Add SUPPORTING SYMPTOMS if needed")
print("5. Check available symptoms: predictor.get_available_symptoms()")
print("\n" + "="*80)
