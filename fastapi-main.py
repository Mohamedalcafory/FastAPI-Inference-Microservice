"""
FastAPI main application with ML inference endpoints.
"""
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.middleware import (
    RequestIDMiddleware,
    PrometheusMiddleware,
    LoggingMiddleware,
)
from app.core.metrics import metrics
from app.api.routes import router as api_router
from app.models.loader import ModelRegistry

logger = structlog.get_logger()

# Global model registry
model_registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting FastAPI Inference Service...")
    
    # Configure logging
    configure_logging()
    
    # Load model
    settings = get_settings()
    await model_registry.load_model(
        backend=settings.MODEL_BACKEND,
        model_path=settings.MODEL_PATH,
        onnx_path=settings.ONNX_PATH,
        warmup_enabled=settings.WARMUP_ENABLED,
        warmup_requests=settings.WARMUP_REQUESTS,
    )
    
    logger.info("Model loaded successfully", 
               backend=model_registry.backend,
               version=model_registry.version)
    
    # Record startup metrics
    metrics.MODEL_LOAD_TIME.observe(model_registry.load_time)
    metrics.MODEL_INFO.info({
        'version': model_registry.version,
        'backend': model_registry.backend,
        'load_time': str(model_registry.load_time)
    })
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI Inference Service...")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="FastAPI ML Inference Service",
        description="Production-ready ML inference microservice with async jobs and monitoring",
        version="0.1.0",
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None,
        lifespan=lifespan,
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Include routers
    app.include_router(api_router, prefix="/api/v1")
    
    return app


# Create app instance
app = create_app()


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "fastapi-inference-service",
        "version": "0.1.0"
    }


@app.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check endpoint."""
    model_ready = model_registry.is_loaded()
    
    return {
        "status": "ready" if model_ready else "not_ready",
        "model_loaded": model_ready,
        "model_backend": model_registry.backend,
        "model_version": model_registry.version,
        "timestamp": time.time()
    }


@app.get("/metrics")
async def get_metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/model/info")
async def model_info() -> Dict[str, Any]:
    """Get model information."""
    if not model_registry.is_loaded():
        return {"error": "Model not loaded"}
    
    return {
        "backend": model_registry.backend,
        "version": model_registry.version,
        "load_time": model_registry.load_time,
        "warmup_completed": model_registry.warmup_completed,
        "feature_names": getattr(model_registry.model, "feature_names_in_", None),
        "n_features": getattr(model_registry.model, "n_features_in_", None),
        "model_type": type(model_registry.model).__name__
    }


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )