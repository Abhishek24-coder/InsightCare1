"""
Quick test script for trained models
Run this after training to verify everything works
"""

from disease_classifier import DiseaseClassifier

def main():
    print("=" * 80)
    print("🧪 QUICK MODEL TEST")
    print("=" * 80)
    
    # Load classifier
    print("\n1. Loading model...")
    classifier = DiseaseClassifier()
    classifier.load_model()
    
    # Show model info
    print("\n2. Model Information:")
    info = classifier.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Interactive testing
    print("\n3. Interactive Testing")
    print("=" * 80)
    print("\nEnter symptoms separated by commas (or 'quit' to exit)")
    print("Example: fever, cough, headache")
    print("\nAvailable symptoms: itching, fever, cough, fatigue, vomiting, etc.")
    
    while True:
        print("\n" + "-" * 80)
        user_input = input("\nSymptoms: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            print("⚠️  Please enter at least one symptom")
            continue
        
        # Parse symptoms
        symptoms = [s.strip() for s in user_input.split(',')]
        
        try:
            # Predict
            result = classifier.predict(symptoms)
            
            # Display results
            print(f"\n🎯 PREDICTION RESULTS")
            print(f"   Disease: {result['disease']}")
            print(f"   Confidence: {result['confidence']*100:.1f}%")
            
            print(f"\n   Top 3 Possibilities:")
            for i, item in enumerate(result['top_3'], 1):
                bar_length = int(item['probability'] * 30)
                bar = '█' * bar_length + '░' * (30 - bar_length)
                print(f"   {i}. {item['disease']:<25} {bar} {item['probability']*100:>5.1f}%")
            
            print(f"\n   💡 Recommendation:")
            print(f"   {result['recommendation']}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Make sure symptoms are spelled correctly and separated by commas")

if __name__ == "__main__":
    main()
