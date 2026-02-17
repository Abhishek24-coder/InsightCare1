"""
Test script to reproduce /api/diagnose endpoint and capture any crashes
"""

import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"


def test_register_and_login():
    """Create test user and get token"""
    # Register
    register_data = {
        "email": "test_diagnose@example.com",
        "password": "TestPassword123!",
        "name": "Test User",  # Changed from full_name to name
    }

    print("1. Attempting registration...")
    try:
        resp = requests.post(f"{BASE_URL}/api/auth/register", json=register_data)
        print(f"   Register status: {resp.status_code}")
        if resp.status_code not in [200, 201, 400]:  # 400 if already exists
            print(f"   Response: {resp.text}")
    except Exception as e:
        print(f"   Register error: {e}")

    # Login
    login_data = {"email": "test_diagnose@example.com", "password": "TestPassword123!"}

    print("\n2. Attempting login...")
    try:
        resp = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
        print(f"   Login status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            token = data.get("access_token")
            print(
                f"   Got token: {token[:50]}..." if token else "   No token in response"
            )
            return token
        else:
            print(f"   Response: {resp.text}")
            return None
    except Exception as e:
        print(f"   Login error: {e}")
        return None


def test_diagnose_endpoint(token):
    """Test the /api/diagnose endpoint"""
    if not token:
        print("\n❌ No token available, skipping diagnose test")
        return

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    diagnose_data = {
        "symptoms": ["fever", "headache", "fatigue"],
        "severity": "moderate",
        "duration": "1-3 days",
    }

    print("\n3. Testing /api/diagnose endpoint...")
    print(f"   Payload: {json.dumps(diagnose_data, indent=2)}")

    try:
        resp = requests.post(
            f"{BASE_URL}/api/diagnose", json=diagnose_data, headers=headers, timeout=10
        )
        print(f"   Status: {resp.status_code}")
        print(f"   Response: {resp.text[:500]}")

        if resp.status_code == 200:
            print("\n✅ /api/diagnose endpoint working!")
        else:
            print(f"\n❌ /api/diagnose failed with status {resp.status_code}")

    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ CONNECTION ERROR - Backend crashed or unreachable")
        print(f"   Details: {e}")
    except requests.exceptions.Timeout:
        print(f"\n❌ TIMEOUT - Backend took too long to respond")
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")


def test_history_endpoint(token):
    """Test the /api/history endpoint"""
    if not token:
        print("\n❌ No token available, skipping history test")
        return

    headers = {"Authorization": f"Bearer {token}"}

    print("\n4. Testing /api/history endpoint...")

    try:
        resp = requests.get(
            f"{BASE_URL}/api/history?page=1&limit=10", headers=headers, timeout=10
        )
        print(f"   Status: {resp.status_code}")
        print(f"   Response: {resp.text[:500]}")

        if resp.status_code == 200:
            print("\n✅ /api/history endpoint working!")
        else:
            print(f"\n❌ /api/history failed with status {resp.status_code}")

    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ CONNECTION ERROR - Backend crashed or unreachable")
        print(f"   Details: {e}")
    except requests.exceptions.Timeout:
        print(f"\n❌ TIMEOUT - Backend took too long to respond")
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing InsightCare Backend Endpoints")
    print("=" * 60)

    # Test health first
    print("\n0. Testing /api/health...")
    try:
        resp = requests.get(f"{BASE_URL}/api/health", timeout=5)
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            print("   ✅ Backend is responsive")
        else:
            print("   ⚠️ Backend responded but not healthy")
    except Exception as e:
        print(f"   ❌ Backend not reachable: {e}")
        sys.exit(1)

    # Get token
    token = test_register_and_login()

    # Test endpoints
    test_diagnose_endpoint(token)
    test_history_endpoint(token)

    print("\n" + "=" * 60)
    print("Test complete - check backend terminal for any crash logs")
    print("=" * 60)
