"""
Interactive Data Explorer
Quick script to check and explore the loaded data
"""

from data_pipeline import DataPipeline
import pandas as pd

def main():
    print("="*70)
    print("DISEASE-SYMPTOM DATA EXPLORER")
    print("="*70)
    
    # Load data
    pipeline = DataPipeline()
    pipeline.load_data()
    df_clean = pipeline.prepare_data()
    symptoms = pipeline.get_unique_symptoms(df_clean)
    diseases = pipeline.get_unique_diseases(df_clean)
    severity_dict = pipeline.create_severity_dict()
    
    # Show available options
    print("\n" + "="*70)
    print("What would you like to check?")
    print("="*70)
    print("\n1. View all diseases (41 total)")
    print("2. View all symptoms (131 total)")
    print("3. View symptoms for a specific disease")
    print("4. View symptom severity weights")
    print("5. View sample data records")
    print("6. Search for diseases by symptom")
    
    while True:
        choice = input("\nEnter choice (1-6) or 'q' to quit: ").strip()
        
        if choice == 'q':
            print("\nGoodbye!")
            break
        
        elif choice == '1':
            print("\n" + "="*70)
            print("ALL DISEASES")
            print("="*70)
            for i, disease in enumerate(diseases, 1):
                print(f"{i:2d}. {disease}")
        
        elif choice == '2':
            print("\n" + "="*70)
            print("ALL SYMPTOMS (with severity weights)")
            print("="*70)
            for i, symptom in enumerate(symptoms, 1):
                severity = severity_dict.get(symptom, 'N/A')
                print(f"{i:3d}. {symptom:40s} (severity: {severity})")
        
        elif choice == '3':
            print("\nAvailable diseases:")
            for i, disease in enumerate(diseases[:10], 1):
                print(f"{i}. {disease}")
            print("...")
            
            disease_name = input("\nEnter disease name (or number 1-10): ").strip()
            
            # Check if it's a number
            try:
                idx = int(disease_name) - 1
                if 0 <= idx < 10:
                    disease_name = diseases[idx]
            except ValueError:
                pass
            
            # Find records with this disease
            disease_records = df_clean[df_clean['Disease'] == disease_name]
            
            if disease_records.empty:
                print(f"\n❌ No records found for '{disease_name}'")
            else:
                print(f"\n" + "="*70)
                print(f"SYMPTOMS FOR: {disease_name}")
                print("="*70)
                
                # Get unique symptoms for this disease
                symptom_cols = [col for col in df_clean.columns if col.startswith('Symptom_')]
                disease_symptoms = set()
                
                for col in symptom_cols:
                    symptoms_in_col = disease_records[col].dropna().unique()
                    disease_symptoms.update(symptoms_in_col)
                
                disease_symptoms = sorted(list(disease_symptoms))
                
                print(f"\nTotal unique symptoms: {len(disease_symptoms)}")
                print(f"Number of records: {len(disease_records)}\n")
                
                for i, symptom in enumerate(disease_symptoms, 1):
                    severity = severity_dict.get(symptom, 'N/A')
                    print(f"{i:2d}. {symptom:40s} (severity: {severity})")
                
                # Show disease info
                info = pipeline.get_disease_info(disease_name)
                if info['description']:
                    print(f"\n📋 Description:")
                    print(f"   {info['description']}")
                
                if info['precautions']:
                    print(f"\n💊 Precautions:")
                    for i, prec in enumerate(info['precautions'], 1):
                        print(f"   {i}. {prec}")
        
        elif choice == '4':
            print("\n" + "="*70)
            print("SYMPTOM SEVERITY WEIGHTS")
            print("="*70)
            print("\nSymptoms grouped by severity:\n")
            
            # Group by severity
            severity_groups = {}
            for symptom, weight in severity_dict.items():
                if weight not in severity_groups:
                    severity_groups[weight] = []
                severity_groups[weight].append(symptom)
            
            for weight in sorted(severity_groups.keys(), reverse=True):
                symptoms_at_weight = severity_groups[weight]
                print(f"\nSeverity {weight} ({len(symptoms_at_weight)} symptoms):")
                for symptom in sorted(symptoms_at_weight):
                    print(f"  • {symptom}")
        
        elif choice == '5':
            print("\n" + "="*70)
            print("SAMPLE DATA RECORDS")
            print("="*70)
            
            # Show first 5 records
            sample = df_clean.head(5)
            
            for idx, row in sample.iterrows():
                print(f"\nRecord {idx + 1}:")
                print(f"  Disease: {row['Disease']}")
                print(f"  Symptoms: ", end="")
                
                symptom_cols = [col for col in df_clean.columns if col.startswith('Symptom_')]
                symptoms_in_record = []
                for col in symptom_cols:
                    if pd.notna(row[col]):
                        symptoms_in_record.append(row[col])
                
                print(", ".join(symptoms_in_record))
        
        elif choice == '6':
            symptom_to_search = input("\nEnter symptom name (e.g., 'fever', 'cough'): ").strip().lower().replace(' ', '_')
            
            print(f"\n" + "="*70)
            print(f"DISEASES WITH SYMPTOM: {symptom_to_search}")
            print("="*70)
            
            # Search in all symptom columns
            symptom_cols = [col for col in df_clean.columns if col.startswith('Symptom_')]
            matching_records = df_clean[
                df_clean[symptom_cols].apply(
                    lambda row: symptom_to_search in row.values, axis=1
                )
            ]
            
            if matching_records.empty:
                print(f"\n❌ No diseases found with symptom '{symptom_to_search}'")
            else:
                diseases_with_symptom = matching_records['Disease'].unique()
                print(f"\nFound {len(diseases_with_symptom)} diseases:\n")
                
                for i, disease in enumerate(sorted(diseases_with_symptom), 1):
                    count = len(matching_records[matching_records['Disease'] == disease])
                    print(f"{i:2d}. {disease:50s} ({count} records)")
        
        else:
            print("\n❌ Invalid choice. Please enter 1-6 or 'q'")


if __name__ == "__main__":
    main()
