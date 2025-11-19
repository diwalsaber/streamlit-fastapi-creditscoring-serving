"""Tests for backend API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_predictor():
    """Mock predictor for testing."""
    predictor = Mock()
    predictor.is_loaded = True
    predictor.is_data_loaded = True
    predictor.data_size = 1000
    predictor.predict_new_client = Mock(return_value=0.25)
    predictor.predict_existing_client = Mock(return_value=0.35)
    predictor.get_risk_level = Mock(return_value="low")
    return predictor


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    @patch('backend.api.routes.predictor')
    def test_predict_new_client(self, mock_pred, client, mock_predictor):
        """Test prediction for new client."""
        mock_pred.configure_mock(**{
            'is_loaded': True,
            'predict_new_client.return_value': 0.25,
            'get_risk_level.return_value': 'low'
        })

        payload = {
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.4,
            "DAYS_BIRTH": -15000,
            "AMT_GOODS_PRICE": 450000.0,
            "AMT_CREDIT": 500000.0,
            "AMT_ANNUITY": 25000.0,
            "DAYS_EMPLOYED": -2000,
            "CODE_GENDER": 1.0,
            "AMT_INCOME_TOTAL": 150000.0,
            "DAYS_EMPLOYED_PERC": 0.13,
            "INCOME_CREDIT_PERC": 0.30,
            "ANNUITY_INCOME_PERC": 0.17,
            "PAYMENT_RATE": 0.05
        }

        response = client.post("/api/v1/predict/new", json=payload)
        assert response.status_code in [200, 503]  # 503 if model not loaded

    def test_predict_new_client_invalid_data(self, client):
        """Test prediction with invalid data."""
        payload = {
            "EXT_SOURCE_1": 2.0,  # Invalid: should be between 0 and 1
        }

        response = client.post("/api/v1/predict/new", json=payload)
        assert response.status_code == 422  # Validation error

    @patch('backend.api.routes.predictor')
    def test_predict_existing_client(self, mock_pred, client):
        """Test prediction for existing client."""
        mock_pred.configure_mock(**{
            'is_loaded': True,
            'is_data_loaded': True,
            'predict_existing_client.return_value': 0.35,
            'get_risk_level.return_value': 'medium'
        })

        payload = {"id_client": 100}
        response = client.post("/api/v1/predict/existing", json=payload)
        assert response.status_code in [200, 503]  # 503 if model not loaded


class TestLegacyEndpoints:
    """Tests for legacy endpoint redirects."""

    def test_legacy_home(self, client):
        """Test legacy home endpoint."""
        response = client.get("/home")
        assert response.status_code == 200
        assert "Use /health" in response.json()["message"]

    def test_legacy_predict_new(self, client):
        """Test legacy predict_new endpoint."""
        response = client.post("/predict_new")
        assert response.status_code == 200
        assert "deprecated" in response.json()["message"]
