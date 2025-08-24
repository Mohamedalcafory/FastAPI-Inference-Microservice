"""
Application configuration using environment variables.
"""
from functools import lru_cache
from typing import List

from decouple import config
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # App Configuration
    DEBUG: bool = config('DEBUG', default=False, cast=bool)
    ENABLE_DOCS: bool = config('ENABLE_DOCS', default=True, cast=bool)
    ALLOWED_ORIGINS: List[str] = config(
        'ALLOWED_ORIGINS', 
        default='http://localhost:3000,http://127.0.0.1:3000', 
        cast=lambda v: [s.strip() for s in v.split(',')]
    )
    
    # Model Configuration
    MODEL_BACKEND: str = config('MODEL_BACKEND', default='sklearn')  # sklearn|onnx
    MODEL_PATH: str = config('MODEL_PATH', default='models/model.joblib')
    ONNX_PATH: str = config('ONNX_PATH', default='models/model.onnx')
    MODEL_VERSION: str = config('MODEL_VERSION', default='1.0.0')
    
    # Warmup Configuration
    WARMUP_ENABLED: bool = config('WARMUP_ENABLED', default=True, cast=bool)
    WARMUP_REQUESTS: int = config('WARMUP_REQUESTS', default=10, cast=int)
    
    # Inference Configuration
    MAX_BATCH_SIZE: int = config('MAX_BATCH_SIZE', default=100, cast=int)
    INFERENCE_TIMEOUT: float = config('INFERENCE_TIMEOUT', default=30.0, cast=float)
    MAX_FEATURES: int = config('MAX_FEATURES', default=1000, cast=int)
    
    # Redis Configuration
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
    
    # Celery Configuration
    CELERY_BROKER_URL: str = config('CELERY_BROKER_URL', default='redis://localhost:6379/1')
    CELERY_RESULT_BACKEND: str = config('CELERY_RESULT_BACKEND', default='redis://localhost:6379/2')
    CELERY_TASK_RESULT_EXPIRES: int = config('CELERY_TASK_RESULT_EXPIRES', default=3600, cast=int)
    
    # Monitoring Configuration
    PROMETHEUS_ENABLED: bool = config('PROMETHEUS_ENABLED', default=True, cast=bool)
    LOG_LEVEL: str = config('LOG_LEVEL', default='INFO')
    REQUEST_ID_HEADER: str = config('REQUEST_ID_HEADER', default='X-Request-ID')
    
    # Performance Configuration
    UVICORN_WORKERS: int = config('UVICORN_WORKERS', default=1, cast=int)
    UVICORN_HOST: str = config('UVICORN_HOST', default='0.0.0.0')
    UVICORN_PORT: int = config('UVICORN_PORT', default=8000, cast=int)
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()