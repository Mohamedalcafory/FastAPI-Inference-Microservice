"""
Pytest configuration and fixtures for testing.
"""
import pytest
import asyncio
from typing import Generator
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.main import app
from app.models.loader import ModelRegistry


@pytest.fixture
def client() -> Generator:
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_model_registry():
    """Mock model registry for testing."""
    registry = Mock(spec=ModelRegistry)
    registry.is_loaded.return_value = True
    registry.backend = "sklearn"
    registry.version = "v1.0.0"
    registry.load_time = 1.0
    registry.warmup_completed = True
    registry.predict.return_value = 1.0
    registry.get_model_info.return_value = {
        "backend": "sklearn",
        "version": "v1.0.0",
        "load_time": 1.0,
        "warmup_completed": True,
        "model_type": "RandomForestClassifier"
    }
    return registry


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request data."""
    return {
        "inputs": [5.1, 3.5, 1.4, 0.2],
        "model_version": None
    }


@pytest.fixture
def sample_batch_request():
    """Sample batch prediction request data."""
    return {
        "batches": [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2]
        ],
        "model_version": None
    }


@pytest.fixture
def sample_async_request():
    """Sample async prediction request data."""
    return {
        "inputs": [5.1, 3.5, 1.4, 0.2],
        "model_version": None
    }


# Async test support
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
