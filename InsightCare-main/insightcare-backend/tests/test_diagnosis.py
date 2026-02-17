# tests/test_diagnosis.py
import pytest
from fastapi import status


class TestDiagnosisAnalyze:
    """Test cases for symptom analysis."""

    def test_analyze_symptoms_success(self, client, auth_headers):
        """Test successful symptom analysis."""
        response = client.post(
            "/api/diagnosis/analyze",
            json={
                "symptoms": ["fever", "cough", "sore throat"],
                "duration_days": 3,
                "severity": "moderate",
            },
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # API returns predictions array with disease predictions
        assert "predictions" in data
        assert "symptoms_analyzed" in data
        assert "disclaimer" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) > 0
        # Check first prediction structure
        pred = data["predictions"][0]
        assert "disease" in pred
        assert "confidence" in pred
        assert isinstance(pred["confidence"], (int, float))
        assert 0 <= pred["confidence"] <= 1  # Confidence is 0-1, not 0-100

    def test_analyze_empty_symptoms(self, client, auth_headers):
        """Test analysis with empty symptoms list."""
        response = client.post(
            "/api/diagnosis/analyze",
            json={"symptoms": [], "duration_days": 1, "severity": "mild"},
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_analyze_no_auth(self, client):
        """Test analysis without authentication."""
        response = client.post(
            "/api/diagnosis/analyze",
            json={"symptoms": ["fever"], "duration_days": 1, "severity": "mild"},
        )
        # Note: Rate limiting middleware may return 403 before auth check returns 401
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        ]

    def test_analyze_invalid_severity(self, client, auth_headers):
        """Test analysis with invalid severity."""
        response = client.post(
            "/api/diagnosis/analyze",
            json={"symptoms": ["fever"], "duration_days": 1, "severity": "invalid"},
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestDiagnosisHistory:
    """Test cases for diagnosis history."""

    def test_get_history_success(self, client, auth_headers):
        """Test getting diagnosis history."""
        # First create a diagnosis
        client.post(
            "/api/diagnosis/analyze",
            json={
                "symptoms": ["fever", "cough"],
                "duration_days": 2,
                "severity": "mild",
            },
            headers=auth_headers,
        )

        # Then get history
        response = client.get("/api/diagnosis/history", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # API returns paginated results with total, page, limit, and results array
        assert "results" in data
        assert "total" in data
        assert "page" in data
        assert "limit" in data
        assert isinstance(data["results"], list)
        if len(data["results"]) > 0:
            diagnosis = data["results"][0]
            assert "symptoms" in diagnosis
            assert "diagnosis_id" in diagnosis
            assert "timestamp" in diagnosis
            assert "top_prediction" in diagnosis

    def test_get_history_no_auth(self, client):
        """Test getting history without authentication."""
        response = client.get("/api/diagnosis/history")
        # Note: Rate limiting middleware may return 403 before auth check returns 401
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        ]

    def test_get_history_pagination(self, client, auth_headers):
        """Test history pagination."""
        # Create multiple diagnoses
        for i in range(5):
            client.post(
                "/api/diagnosis/analyze",
                json={
                    "symptoms": ["fever"],
                    "duration_days": i + 1,
                    "severity": "mild",
                },
                headers=auth_headers,
            )

        # Get first page
        response = client.get(
            "/api/diagnosis/history?limit=2&page=1", headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert len(data["results"]) <= 2
        assert data["limit"] == 2
        assert data["page"] == 1

        # Get second page
        response = client.get(
            "/api/diagnosis/history?limit=2&page=2", headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data2 = response.json()
        assert "results" in data2
        assert data2["page"] == 2
