"""
Test API Endpoints
"""

import pytest
from fastapi.testclient import TestClient


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_readiness_check(client):
    """Test readiness check endpoint"""
    response = client.get("/ready")
    assert response.status_code in [200, 503]  # May not be ready if models not loaded


def test_api_info(client, auth_headers):
    """Test API info endpoint"""
    response = client.get("/v2/info", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "api_version" in data
    assert "pipeline_stages" in data


def test_extract_without_auth(client):
    """Test extraction without authentication"""
    response = client.post("/v2/extract/kk")
    assert response.status_code == 401


def test_extract_with_invalid_file(client, auth_headers):
    """Test extraction with invalid file"""
    files = {"file": ("test.txt", b"invalid content", "text/plain")}
    response = client.post("/v2/extract/kk", headers=auth_headers, files=files)
    assert response.status_code == 400


# Note: Full integration test requires model files
# Add more comprehensive tests once models are available
