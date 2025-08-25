"""
Configuration management for the FastAPI inference service.
"""
from typing import List
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Model Configuration
    MODEL_BACKEND: str = Field(default="sklearn", description="Model backend: sklearn or onnx")
    MODEL_PATH: str = Field(default="models/model.joblib", description="Path to sklearn model")
    ONNX_PATH: str = Field(default="models/model.onnx", description="Path to ONNX model")
    WARMUP_ENABLED: bool = Field(default=True, description="Enable model warmup")
    WARMUP_REQUESTS: int = Field(default=10, description="Number of warmup requests")
    
    # Performance Settings
    MAX_BATCH_SIZE: int = Field(default=100, description="Maximum batch size for predictions")
    INFERENCE_TIMEOUT: float = Field(default=30.0, description="Inference timeout in seconds")
    UVICORN_WORKERS: int = Field(default=1, description="Number of Uvicorn workers")
    
    # Redis/Celery Configuration
    REDIS_URL: str = Field(default="redis://redis:6379/0", description="Redis URL")
    CELERY_BROKER_URL: str = Field(default="redis://redis:6379/1", description="Celery broker URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://redis:6379/2", description="Celery result backend")
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = Field(default=True, description="Enable Prometheus metrics")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Application Settings
    DEBUG: bool = Field(default=False, description="Debug mode")
    ENABLE_DOCS: bool = Field(default=True, description="Enable API documentation")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins"
    )
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get application settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
