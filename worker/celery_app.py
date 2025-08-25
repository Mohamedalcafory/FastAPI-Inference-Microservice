"""
Celery worker for async prediction tasks.
"""
import os
import sys
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.models.loader import ModelRegistry
from app.services.tasks import celery_app

# Configure logging
configure_logging()
logger = get_logger("celery_worker")

# Global model registry
model_registry = None


@celery_app.on_after_configure.connect
def setup_worker(sender, **kwargs):
    """Setup worker after configuration."""
    logger.info("Setting up Celery worker...")
    
    # Load model in worker
    global model_registry
    settings = get_settings()
    
    model_registry = ModelRegistry()
    
    # Load model synchronously in worker
    import asyncio
    try:
        asyncio.run(model_registry.load_model(
            backend=settings.MODEL_BACKEND,
            model_path=settings.MODEL_PATH,
            onnx_path=settings.ONNX_PATH,
            warmup_enabled=settings.WARMUP_ENABLED,
            warmup_requests=settings.WARMUP_REQUESTS,
        ))
        logger.info("Model loaded in worker", backend=model_registry.backend)
    except Exception as e:
        logger.error("Failed to load model in worker", error=str(e))
        raise


@celery_app.task
def health_check():
    """Health check task."""
    return {
        "status": "healthy",
        "model_loaded": model_registry.is_loaded() if model_registry else False,
        "backend": model_registry.backend if model_registry else None
    }


if __name__ == "__main__":
    # Run Celery worker
    celery_app.start()
