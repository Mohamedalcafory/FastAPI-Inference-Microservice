"""
API routes for the FastAPI inference service.
"""
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import get_logger, log_inference
from app.core.metrics import metrics
from app.models.loader import ModelRegistry
from app.schemas.request import (
    PredictionRequest,
    BatchPredictionRequest,
    AsyncPredictionRequest
)
from app.schemas.response import (
    PredictionResponse,
    BatchPredictionResponse,
    AsyncTaskResponse,
    AsyncTaskStatusResponse,
    ModelInfoResponse,
    ErrorResponse
)
from app.services.tasks import celery_app, async_predict_task

# Create router
router = APIRouter()

# Get logger
logger = get_logger("api_routes")

# Global model registry (will be injected)
model_registry: ModelRegistry = None


def get_model_registry() -> ModelRegistry:
    """Dependency to get model registry."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_registry


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    req: Request,
    registry: ModelRegistry = Depends(get_model_registry)
) -> PredictionResponse:
    """Single prediction endpoint."""
    start_time = time.time()
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        # Validate model version if specified
        if request.model_version and request.model_version != registry.version:
            raise HTTPException(
                status_code=400,
                detail=f"Model version {request.model_version} not available. Available: {registry.version}"
            )
        
        # Make prediction
        inference_start = time.time()
        prediction = registry.predict(request.inputs)
        inference_duration = (time.time() - inference_start) * 1000
        
        # Calculate total duration
        total_duration = (time.time() - start_time) * 1000
        
        # Record metrics
        metrics.MODEL_PREDICTIONS.labels(
            backend=registry.backend,
            endpoint_type="single"
        ).inc()
        
        metrics.MODEL_INFERENCE_DURATION.labels(
            backend=registry.backend,
            batch_size=1
        ).observe(inference_duration / 1000)
        
        # Log inference
        log_inference(
            request_id=request_id,
            backend=registry.backend,
            batch_size=1,
            duration_ms=inference_duration
        )
        
        return PredictionResponse(
            prediction=prediction,
            model_version=registry.version,
            backend=registry.backend,
            inference_duration_ms=inference_duration,
            total_duration_ms=total_duration
        )
        
    except Exception as e:
        logger.error(
            "Prediction failed",
            request_id=request_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    req: Request,
    registry: ModelRegistry = Depends(get_model_registry)
) -> BatchPredictionResponse:
    """Batch prediction endpoint."""
    start_time = time.time()
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        # Validate model version if specified
        if request.model_version and request.model_version != registry.version:
            raise HTTPException(
                status_code=400,
                detail=f"Model version {request.model_version} not available. Available: {registry.version}"
            )
        
        # Make predictions
        inference_start = time.time()
        predictions = registry.predict(request.batches)
        inference_duration = (time.time() - inference_start) * 1000
        
        # Calculate total duration
        total_duration = (time.time() - start_time) * 1000
        
        # Record metrics
        batch_size = len(request.batches)
        metrics.MODEL_PREDICTIONS.labels(
            backend=registry.backend,
            endpoint_type="batch"
        ).inc()
        
        metrics.MODEL_INFERENCE_DURATION.labels(
            backend=registry.backend,
            batch_size=batch_size
        ).observe(inference_duration / 1000)
        
        # Log inference
        log_inference(
            request_id=request_id,
            backend=registry.backend,
            batch_size=batch_size,
            duration_ms=inference_duration
        )
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=batch_size,
            model_version=registry.version,
            backend=registry.backend,
            inference_duration_ms=inference_duration,
            total_duration_ms=total_duration
        )
        
    except Exception as e:
        logger.error(
            "Batch prediction failed",
            request_id=request_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/async", response_model=AsyncTaskResponse)
async def predict_async(
    request: AsyncPredictionRequest,
    req: Request,
    registry: ModelRegistry = Depends(get_model_registry)
) -> AsyncTaskResponse:
    """Async prediction endpoint."""
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        # Validate model version if specified
        if request.model_version and request.model_version != registry.version:
            raise HTTPException(
                status_code=400,
                detail=f"Model version {request.model_version} not available. Available: {registry.version}"
            )
        
        # Determine input type
        input_type = "batch" if isinstance(request.inputs[0], list) else "single"
        
        # Submit async task
        task = async_predict_task.delay(
            inputs=request.inputs,
            model_version=request.model_version,
            input_type=input_type
        )
        
        # Record metrics
        metrics.ASYNC_TASKS.labels(
            status="submitted",
            task_type=input_type
        ).inc()
        
        logger.info(
            "Async prediction submitted",
            request_id=request_id,
            task_id=task.id,
            input_type=input_type
        )
        
        return AsyncTaskResponse(
            task_id=task.id,
            status="submitted",
            estimated_duration_seconds=5.0  # Rough estimate
        )
        
    except Exception as e:
        logger.error(
            "Async prediction submission failed",
            request_id=request_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict/async/{task_id}", response_model=AsyncTaskStatusResponse)
async def get_async_status(
    task_id: str,
    req: Request
) -> AsyncTaskStatusResponse:
    """Get async task status."""
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        # Get task result
        task_result = celery_app.AsyncResult(task_id)
        
        if task_result.state == "PENDING":
            return AsyncTaskStatusResponse(
                task_id=task_id,
                status="pending",
                progress=0,
                created_at=task_result.date_done or task_result.date_done
            )
        
        elif task_result.state == "PROGRESS":
            meta = task_result.info or {}
            return AsyncTaskStatusResponse(
                task_id=task_id,
                status="processing",
                progress=meta.get("progress", 0),
                created_at=task_result.date_done or task_result.date_done
            )
        
        elif task_result.state == "SUCCESS":
            result = task_result.result
            return AsyncTaskStatusResponse(
                task_id=task_id,
                status="completed",
                progress=100,
                result=result.get("result"),
                created_at=result.get("created_at"),
                completed_at=result.get("completed_at"),
                model_version=result.get("model_version"),
                backend=result.get("backend")
            )
        
        elif task_result.state == "FAILURE":
            return AsyncTaskStatusResponse(
                task_id=task_id,
                status="failed",
                error=str(task_result.info),
                created_at=task_result.date_done or task_result.date_done
            )
        
        else:
            return AsyncTaskStatusResponse(
                task_id=task_id,
                status=task_result.state,
                created_at=task_result.date_done or task_result.date_done
            )
            
    except Exception as e:
        logger.error(
            "Failed to get task status",
            request_id=request_id,
            task_id=task_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/predict/async/{task_id}")
async def cancel_async_task(
    task_id: str,
    req: Request
) -> Dict[str, Any]:
    """Cancel async task."""
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        # Revoke task
        celery_app.control.revoke(task_id, terminate=True)
        
        logger.info(
            "Task cancelled",
            request_id=request_id,
            task_id=task_id
        )
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task cancelled successfully"
        }
        
    except Exception as e:
        logger.error(
            "Failed to cancel task",
            request_id=request_id,
            task_id=task_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(
    req: Request,
    registry: ModelRegistry = Depends(get_model_registry)
) -> ModelInfoResponse:
    """Get model information."""
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        info = registry.get_model_info()
        
        return ModelInfoResponse(
            backend=info["backend"],
            version=info["version"],
            load_time=info["load_time"],
            warmup_completed=info["warmup_completed"],
            feature_names=info.get("feature_names"),
            n_features=info.get("n_features"),
            model_type=info["model_type"]
        )
        
    except Exception as e:
        logger.error(
            "Failed to get model info",
            request_id=request_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


# Global exception handler
@router.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        "Unhandled exception",
        request_id=request_id,
        error=str(exc),
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_type=type(exc).__name__,
            request_id=request_id
        ).dict()
    )
