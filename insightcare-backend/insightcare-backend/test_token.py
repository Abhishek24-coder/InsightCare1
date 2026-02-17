#!/usr/bin/env python3
"""Test JWT token generation and verification"""
import uuid
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.utils.security import create_access_token, verify_token

# Test with a UUID user_id (like in production)
test_user_id = str(uuid.uuid4())
test_email = "test@example.com"

print("🔐 Testing JWT Token System")
print("=" * 50)
print(f"Test User ID: {test_user_id}")
print(f"Test Email: {test_email}")
print()

# Create token
token_data = {"user_id": test_user_id, "email": test_email}
token = create_access_token(token_data)
print(f"✅ Token Created:")
print(f"   {token[:50]}...")
print()

# Verify token
payload = verify_token(token)
if payload:
    print("✅ Token Verified Successfully!")
    print(f"   user_id: {payload.get('user_id')}")
    print(f"   email: {payload.get('email')}")
    print(f"   exp: {payload.get('exp')}")

    # Verify user_id can be converted to UUID
    try:
        retrieved_uuid = uuid.UUID(payload.get("user_id"))
        print(f"   ✅ UUID conversion successful: {retrieved_uuid}")
    except Exception as e:
        print(f"   ❌ UUID conversion failed: {e}")
else:
    print("❌ Token verification failed!")

print()
print("=" * 50)
print("✅ Token system test complete!")
