"""
Tests for sync inference endpoints.
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["service"] == "fastapi-inference-service"
    assert data["version"] == "0.1.0"


def test_ready_endpoint(client: TestClient):
    """Test readiness check endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data


def test_model_info_endpoint(client: TestClient):
    """Test model info endpoint."""
    response = client.get("/model/info")
    assert response.status_code == 200
    
    data = response.json()
    # Model might not be loaded in test environment
    if "error" not in data:
        assert "backend" in data
        assert "version" in data
        assert "load_time" in data
        assert "warmup_completed" in data


def test_metrics_endpoint(client: TestClient):
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


@patch('app.api.routes.model_registry')
def test_single_prediction(mock_registry, client: TestClient, sample_prediction_request):
    """Test single prediction endpoint."""
    # Mock model registry
    mock_registry.is_loaded.return_value = True
    mock_registry.backend = "sklearn"
    mock_registry.version = "v1.0.0"
    mock_registry.predict.return_value = 1.0
    
    response = client.post("/api/v1/predict", json=sample_prediction_request)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "model_version" in data
    assert "backend" in data
    assert "inference_duration_ms" in data
    assert "total_duration_ms" in data
    assert data["prediction"] == 1.0


@patch('app.api.routes.model_registry')
def test_batch_prediction(mock_registry, client: TestClient, sample_batch_request):
    """Test batch prediction endpoint."""
    # Mock model registry
    mock_registry.is_loaded.return_value = True
    mock_registry.backend = "sklearn"
    mock_registry.version = "v1.0.0"
    mock_registry.predict.return_value = [1.0, 0.0]
    
    response = client.post("/api/v1/predict/batch", json=sample_batch_request)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert "batch_size" in data
    assert "model_version" in data
    assert "backend" in data
    assert "inference_duration_ms" in data
    assert "total_duration_ms" in data
    assert data["batch_size"] == 2
    assert data["predictions"] == [1.0, 0.0]


def test_single_prediction_validation_error(client: TestClient):
    """Test single prediction with invalid input."""
    invalid_request = {
        "inputs": [],  # Empty inputs
        "model_version": None
    }
    
    response = client.post("/api/v1/predict", json=invalid_request)
    assert response.status_code == 422  # Validation error


def test_batch_prediction_validation_error(client: TestClient):
    """Test batch prediction with invalid input."""
    invalid_request = {
        "batches": [],  # Empty batches
        "model_version": None
    }
    
    response = client.post("/api/v1/predict/batch", json=invalid_request)
    assert response.status_code == 422  # Validation error


@patch('app.api.routes.model_registry')
def test_model_not_loaded_error(mock_registry, client: TestClient, sample_prediction_request):
    """Test prediction when model is not loaded."""
    # Mock model registry to return not loaded
    mock_registry.is_loaded.return_value = False
    
    response = client.post("/api/v1/predict", json=sample_prediction_request)
    assert response.status_code == 503  # Service unavailable


def test_invalid_model_version(client: TestClient, sample_prediction_request):
    """Test prediction with invalid model version."""
    request_with_version = {
        **sample_prediction_request,
        "model_version": "invalid_version"
    }
    
    response = client.post("/api/v1/predict", json=request_with_version)
    # Should return 400 if model version doesn't match
    assert response.status_code in [400, 503]
