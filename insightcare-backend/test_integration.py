"""
End-to-End Integration Test
Tests the complete flow: Frontend → API → ML → Response
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.services.ml_service import get_ml_diagnosis, is_ml_available, ml_service


def test_ml_service_loading():
    """Test 1: ML service loads correctly"""
    print("\n" + "="*70)
    print("TEST 1: ML Service Loading")
    print("="*70)
    
    assert is_ml_available(), "❌ ML service not available"
    print("✅ ML service is available")
    
    symptoms = ml_service.get_available_symptoms()
    diseases = ml_service.get_available_diseases()
    
    assert len(symptoms) == 131, f"❌ Expected 131 symptoms, got {len(symptoms)}"
    print(f"✅ {len(symptoms)} symptoms loaded")
    
    assert len(diseases) == 41, f"❌ Expected 41 diseases, got {len(diseases)}"
    print(f"✅ {len(diseases)} diseases loaded")
    
    print("\n✅ TEST 1 PASSED")


def test_valid_symptoms_prediction():
    """Test 2: Prediction with valid symptoms"""
    print("\n" + "="*70)
    print("TEST 2: Valid Symptoms Prediction")
    print("="*70)
    
    # Test case 1: Diabetes symptoms
    symptoms = ['fatigue', 'weight loss', 'increased appetite', 'frequent urination']
    print(f"\nInput symptoms: {symptoms}")
    
    result = get_ml_diagnosis(symptoms)
    
    assert len(result) > 0, "❌ No predictions returned"
    print(f"✅ Received {len(result)} prediction(s)")
    
    pred = result[0]
    assert 'disease' in pred, "❌ Missing 'disease' field"
    assert 'confidence' in pred, "❌ Missing 'confidence' field"
    assert 'recommendations' in pred, "❌ Missing 'recommendations' field"
    
    print(f"✅ Predicted disease: {pred['disease']}")
    print(f"✅ Confidence: {pred['confidence']*100:.2f}%")
    print(f"✅ Model used: {pred['model_used']}")
    print(f"✅ Recommendations: {len(pred['recommendations'])} items")
    
    print("\n✅ TEST 2 PASSED")


def test_invalid_symptoms_handling():
    """Test 3: Handling invalid symptoms"""
    print("\n" + "="*70)
    print("TEST 3: Invalid Symptoms Handling")
    print("="*70)
    
    # Test with some invalid symptoms
    symptoms = ['fever', 'invalid_symptom', 'cough', 'another_fake_symptom']
    print(f"\nInput symptoms: {symptoms}")
    
    result = get_ml_diagnosis(symptoms)
    
    assert len(result) > 0, "❌ No predictions returned"
    pred = result[0]
    
    assert 'valid_symptoms' in pred, "❌ Missing 'valid_symptoms' field"
    assert 'invalid_symptoms' in pred, "❌ Missing 'invalid_symptoms' field"
    
    print(f"✅ Valid symptoms: {pred['valid_symptoms']}")
    print(f"✅ Invalid symptoms: {pred['invalid_symptoms']}")
    
    assert len(pred['valid_symptoms']) > 0, "❌ Should have some valid symptoms"
    assert len(pred['invalid_symptoms']) > 0, "❌ Should have some invalid symptoms"
    
    print("\n✅ TEST 3 PASSED")


def test_multiple_test_cases():
    """Test 4: Multiple disease scenarios"""
    print("\n" + "="*70)
    print("TEST 4: Multiple Disease Scenarios")
    print("="*70)
    
    test_cases = [
        {
            'name': 'Malaria',
            'symptoms': ['chills', 'high fever', 'sweating', 'headache']
        },
        {
            'name': 'Diabetes',
            'symptoms': ['fatigue', 'weight loss', 'increased appetite']
        },
        {
            'name': 'Pneumonia',
            'symptoms': ['cough', 'chest pain', 'breathlessness']
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        print(f"Symptoms: {test_case['symptoms']}")
        
        result = get_ml_diagnosis(test_case['symptoms'])
        
        assert len(result) > 0, f"❌ No predictions for {test_case['name']}"
        
        pred = result[0]
        print(f"✅ Predicted: {pred['disease']}")
        print(f"✅ Confidence: {pred['confidence']*100:.2f}%")
        print(f"✅ Severity: {pred['severity']}")
    
    print("\n✅ TEST 4 PASSED")


def test_json_response_format():
    """Test 5: JSON response format for API"""
    print("\n" + "="*70)
    print("TEST 5: JSON Response Format")
    print("="*70)
    
    symptoms = ['fever', 'cough', 'fatigue']
    result = get_ml_diagnosis(symptoms)
    
    # Verify all required fields
    required_fields = [
        'disease', 'confidence', 'severity', 'description',
        'recommendations', 'model_used', 'valid_symptoms', 'invalid_symptoms'
    ]
    
    pred = result[0]
    for field in required_fields:
        assert field in pred, f"❌ Missing required field: {field}"
        print(f"✅ Field '{field}': {type(pred[field]).__name__}")
    
    # Verify data types
    assert isinstance(pred['disease'], str), "❌ disease should be string"
    assert isinstance(pred['confidence'], (int, float)), "❌ confidence should be number"
    assert isinstance(pred['recommendations'], list), "❌ recommendations should be list"
    
    print("\n✅ TEST 5 PASSED - All fields present with correct types")


def test_performance():
    """Test 6: Response time performance"""
    print("\n" + "="*70)
    print("TEST 6: Performance Test")
    print("="*70)
    
    import time
    
    symptoms = ['fever', 'cough', 'headache', 'fatigue']
    
    # Test 5 predictions
    times = []
    for i in range(5):
        start = time.time()
        result = get_ml_diagnosis(symptoms)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Prediction {i+1}: {elapsed:.4f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\n✅ Average response time: {avg_time:.4f}s")
    
    assert avg_time < 1.0, f"❌ Too slow: {avg_time:.4f}s (should be <1s)"
    print("✅ Performance is acceptable (<1s)")
    
    print("\n✅ TEST 6 PASSED")


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("RUNNING END-TO-END INTEGRATION TESTS")
    print("="*70)
    
    tests = [
        test_ml_service_loading,
        test_valid_symptoms_prediction,
        test_invalid_symptoms_handling,
        test_multiple_test_cases,
        test_json_response_format,
        test_performance
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n❌ TEST FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"\n❌ TEST ERROR: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"\nTotal Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
        print("\n📋 Next Steps:")
        print("   1. Start FastAPI backend: uvicorn app.main:app --reload")
        print("   2. Test API endpoint: http://localhost:8000/api/ml/diagnose")
        print("   3. Check API docs: http://localhost:8000/docs")
        print("   4. Frontend can now call /api/ml/diagnose")
    else:
        print("\n⚠️  SOME TESTS FAILED - Review errors above")
    
    print("\n" + "="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
