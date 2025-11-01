"""
Test Configuration
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers fixture"""
    return {
        "Authorization": "Bearer dev-token"
    }


@pytest.fixture
def sample_image_path():
    """Sample image path fixture"""
    return "tests/fixtures/sample_kk.jpg"
