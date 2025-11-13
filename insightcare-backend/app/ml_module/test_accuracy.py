"""
Test Model Accuracy with Real Cases
"""

from predict import DiseasePredictor

print("\n" + "="*70)
print("TESTING MODEL ACCURACY")
print("="*70)

predictor = DiseasePredictor()

# Test cases with known diseases
test_cases = [
    {
        'symptoms': ['itching', 'skin_rash', 'nodal_skin_eruptions'],
        'expected': 'Fungal infection'
    },
    {
        'symptoms': ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'],
        'expected': 'Allergy'
    },
    {
        'symptoms': ['stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting', 'cough'],
        'expected': 'GERD'
    },
    {
        'symptoms': ['fatigue', 'weight_loss', 'restlessness', 'lethargy', 'irregular_sugar_level', 'increased_appetite', 'polyuria'],
        'expected': 'Diabetes'
    },
    {
        'symptoms': ['high_fever', 'headache', 'nausea', 'muscle_pain', 'chills', 'sweating'],
        'expected': 'Malaria'
    },
    {
        'symptoms': ['joint_pain', 'vomiting', 'fatigue', 'high_fever', 'headache', 'nausea', 'loss_of_appetite'],
        'expected': 'Dengue'
    },
    {
        'symptoms': ['back_pain', 'weakness_in_limbs', 'neck_pain', 'dizziness', 'loss_of_balance'],
        'expected': 'Cervical spondylosis'
    },
    {
        'symptoms': ['cough', 'high_fever', 'breathlessness', 'sweating', 'chest_pain'],
        'expected': 'Pneumonia'
    }
]

correct = 0
total = len(test_cases)

for i, case in enumerate(test_cases, 1):
    print(f"\n{'-'*70}")
    print(f"Test Case {i}: {case['expected']}")
    print(f"{'-'*70}")
    print(f"Symptoms: {', '.join(case['symptoms'])}")
    
    result = predictor.predict(case['symptoms'])
    predicted = result['predicted_disease']
    confidence = result['confidence_percentage']
    
    is_correct = predicted.lower() == case['expected'].lower()
    if is_correct:
        correct += 1
    
    status = '✅ CORRECT' if is_correct else '❌ WRONG'
    
    print(f"\nExpected:  {case['expected']}")
    print(f"Predicted: {predicted} ({confidence}%)")
    print(f"Status:    {status}")
    
    if not is_correct:
        print(f"\n🔍 Top 3 Predictions:")
        top_3 = result.get('top_predictions', [])[:3]
        for j, pred in enumerate(top_3, 1):
            print(f"   {j}. {pred['disease']:<30} {pred['confidence']:.1f}%")

print("\n" + "="*70)
print(f"FINAL ACCURACY: {correct}/{total} = {(correct/total)*100:.1f}%")
print("="*70)

if correct < total:
    print(f"\n⚠️  {total - correct} predictions were incorrect")
    print("\n💡 Tips to improve accuracy:")
    print("   • Add more specific symptoms")
    print("   • Use exact symptom names from the dataset")
    print("   • Include severity information")
    print("   • Try with 5-7 symptoms for better results")
else:
    print("\n🎉 Perfect accuracy! All predictions correct!")
