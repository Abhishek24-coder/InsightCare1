# tests/test_auth.py
import pytest
from fastapi import status


class TestUserRegistration:
    """Test cases for user registration."""

    def test_register_success(self, client):
        """Test successful user registration."""
        response = client.post(
            "/api/auth/register",
            json={
                "name": "New User",
                "email": "newuser@example.com",
                "password": "Password123!",
            },
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "message" in data
        assert data["user"]["email"] == "newuser@example.com"
        assert data["user"]["name"] == "New User"
        assert "password" not in data["user"]

    def test_register_duplicate_email(self, client, test_user):
        """Test registration with existing email."""
        response = client.post(
            "/api/auth/register",
            json={
                "name": "Duplicate",
                "email": test_user.email,
                "password": "Password123!",
            },
        )
        assert response.status_code == status.HTTP_409_CONFLICT

    def test_register_invalid_email(self, client):
        """Test registration with invalid email."""
        response = client.post(
            "/api/auth/register",
            json={"name": "User", "email": "invalid-email", "password": "Password123!"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_register_short_password(self, client):
        """Test registration with password too short."""
        response = client.post(
            "/api/auth/register",
            json={"name": "User", "email": "user@example.com", "password": "123"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_register_password_without_number(self, client):
        """Test registration with password without number."""
        response = client.post(
            "/api/auth/register",
            json={"name": "User", "email": "user@example.com", "password": "Password"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestUserLogin:
    """Test cases for user login."""

    def test_login_success(self, client, test_user):
        """Test successful login."""
        response = client.post(
            "/api/auth/login", json={"email": test_user.email, "password": "Test123!"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["user"]["email"] == test_user.email

    def test_login_wrong_password(self, client, test_user):
        """Test login with wrong password."""
        response = client.post(
            "/api/auth/login",
            json={"email": test_user.email, "password": "WrongPassword123!"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_login_nonexistent_user(self, client):
        """Test login with non-existent email."""
        response = client.post(
            "/api/auth/login",
            json={"email": "nonexistent@example.com", "password": "Password123!"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestUserProfile:
    """Test cases for user profile."""

    def test_get_profile_success(self, client, auth_headers):
        """Test getting user profile with valid token."""
        response = client.get("/api/auth/me", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "email" in data
        assert "name" in data

    def test_get_profile_no_token(self, client):
        """Test getting profile without authentication."""
        response = client.get("/api/auth/me")
        # Note: Rate limiting middleware may return 403 before auth check returns 401
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        ]

    def test_get_profile_invalid_token(self, client):
        """Test getting profile with invalid token."""
        response = client.get(
            "/api/auth/me", headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
