"""
Celery tasks for async prediction processing.
"""
import time
from typing import Any, Dict, List, Union
from datetime import datetime

from celery import Celery
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import metrics
from app.models.loader import ModelRegistry

# Get settings
settings = get_settings()

# Create Celery app
celery_app = Celery(
    "inference_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.services.tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.INFERENCE_TIMEOUT,
    task_soft_time_limit=settings.INFERENCE_TIMEOUT - 5,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
)

# Global model registry (will be initialized in worker)
model_registry = None
logger = get_logger("celery_tasks")


def get_model_registry() -> ModelRegistry:
    """Get or create model registry instance."""
    global model_registry
    if model_registry is None:
        model_registry = ModelRegistry()
        # Note: In a real implementation, you'd want to load the model here
        # For now, we'll assume it's already loaded by the main app
    return model_registry


@celery_app.task(bind=True)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def async_predict_task(
    self,
    inputs: Union[List[float], List[List[float]]],
    model_version: str = None,
    input_type: str = "single"
) -> Dict[str, Any]:
    """Async prediction task with retries."""
    start_time = time.time()
    task_id = self.request.id
    
    try:
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={
                "status": "processing",
                "progress": 0,
                "task_id": task_id
            }
        )
        
        # Get model registry
        registry = get_model_registry()
        
        # Validate model is loaded
        if not registry.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={
                "status": "processing",
                "progress": 50,
                "task_id": task_id
            }
        )
        
        # Make prediction
        inference_start = time.time()
        predictions = registry.predict(inputs)
        inference_duration = (time.time() - inference_start) * 1000
        
        # Calculate total duration
        total_duration = (time.time() - start_time) * 1000
        
        # Record metrics
        batch_size = len(inputs) if isinstance(inputs[0], list) else 1
        metrics.MODEL_PREDICTIONS.labels(
            backend=registry.backend,
            endpoint_type="async"
        ).inc()
        
        metrics.MODEL_INFERENCE_DURATION.labels(
            backend=registry.backend,
            batch_size=batch_size
        ).observe(inference_duration / 1000)
        
        metrics.ASYNC_TASKS.labels(
            status="completed",
            task_type=input_type
        ).inc()
        
        metrics.ASYNC_TASK_DURATION.labels(
            task_type=input_type,
            status="completed"
        ).observe(total_duration / 1000)
        
        # Log completion
        logger.info(
            "Async prediction completed",
            task_id=task_id,
            backend=registry.backend,
            batch_size=batch_size,
            inference_duration_ms=inference_duration,
            total_duration_ms=total_duration
        )
        
        # Return result
        result = {
            "task_id": task_id,
            "status": "completed",
            "progress": 100,
            "result": predictions,
            "model_version": registry.version,
            "backend": registry.backend,
            "inference_duration_ms": inference_duration,
            "total_duration_ms": total_duration,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": datetime.utcnow().isoformat()
        }
        
        return result
        
    except Exception as e:
        # Record error metrics
        metrics.ASYNC_TASKS.labels(
            status="failed",
            task_type=input_type
        ).inc()
        
        metrics.ASYNC_TASK_DURATION.labels(
            task_type=input_type,
            status="failed"
        ).observe((time.time() - start_time))
        
        # Log error
        logger.error(
            "Async prediction failed",
            task_id=task_id,
            error=str(e),
            duration_ms=(time.time() - start_time) * 1000
        )
        
        # Re-raise for retry
        raise


@celery_app.task
def health_check_task() -> Dict[str, Any]:
    """Health check task for Celery worker."""
    try:
        registry = get_model_registry()
        return {
            "status": "healthy",
            "model_loaded": registry.is_loaded(),
            "backend": registry.backend,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task
def update_queue_metrics() -> None:
    """Update queue depth metrics."""
    try:
        # Get queue statistics
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        reserved_tasks = inspect.reserved()
        
        if active_tasks:
            for worker, tasks in active_tasks.items():
                metrics.CELERY_QUEUE_DEPTH.labels(queue_name=worker).set(len(tasks))
        
        if reserved_tasks:
            for worker, tasks in reserved_tasks.items():
                current_depth = metrics.CELERY_QUEUE_DEPTH.labels(queue_name=worker)._value.get()
                metrics.CELERY_QUEUE_DEPTH.labels(queue_name=worker).set(current_depth + len(tasks))
                
    except Exception as e:
        logger.warning("Failed to update queue metrics", error=str(e))
