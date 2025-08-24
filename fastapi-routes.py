"""
API routes for ML inference endpoints.
"""
import time
from typing import List, Dict, Any, Optional
import uuid

import structlog
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.metrics import metrics
from app.schemas.request import (
    PredictionRequest,
    BatchPredictionRequest,
    AsyncPredictionRequest
)
from app.schemas.response import (
    PredictionResponse,
    BatchPredictionResponse,
    AsyncTaskResponse,
    AsyncTaskStatusResponse
)
from app.models.predict import InferenceService
from app.services.tasks import async_predict_task
from worker.celery_app import celery_app

logger = structlog.get_logger()
router = APIRouter()

# Global inference service
inference_service = InferenceService()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    settings = Depends(get_settings)
) -> PredictionResponse:
    """
    Synchronous single prediction endpoint.
    
    - **inputs**: Feature vector or array of features
    - **model_version**: Optional model version (for A/B testing)
    """
    start_time = time.time()
    
    try:
        # Validate input size
        if len(request.inputs) > settings.MAX_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Too many features. Maximum allowed: {settings.MAX_FEATURES}"
            )
        
        # Make prediction
        prediction_start = time.time()
        prediction = await inference_service.predict(
            inputs=request.inputs,
            model_version=request.model_version
        )
        inference_duration = time.time() - prediction_start
        
        # Record metrics
        metrics.PREDICTION_COUNT.labels(
            backend=inference_service.backend,
            endpoint_type='sync'
        ).inc()
        
        metrics.INFERENCE_LATENCY.labels(
            backend=inference_service.backend,
            batch_size='1'
        ).observe(inference_duration)
        
        total_duration = time.time() - start_time
        
        logger.info(
            "Prediction completed",
            inference_duration_ms=inference_duration * 1000,
            total_duration_ms=total_duration * 1000,
            backend=inference_service.backend
        )
        
        return PredictionResponse(
            prediction=prediction,
            model_version=inference_service.version,
            backend=inference_service.backend,
            inference_duration_ms=inference_duration * 1000,
            total_duration_ms=total_duration * 1000
        )
        
    except Exception as e:
        metrics.ERROR_COUNT.labels(
            error_type=type(e).__name__,
            endpoint='predict'
        ).inc()
        
        logger.error("Prediction failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    settings = Depends(get_settings)
) -> BatchPredictionResponse:
    """
    Synchronous batch prediction endpoint.
    
    - **batches**: List of feature vectors
    - **model_version**: Optional model version
    """
    start_time = time.time()
    
    try:
        # Validate batch size
        if len(request.batches) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Batch too large. Maximum size: {settings.MAX_BATCH_SIZE}"
            )
        
        # Validate feature count
        for i, batch in enumerate(request.batches):
            if len(batch) > settings.MAX_FEATURES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch {i} has too many features. Maximum: {settings.MAX_FEATURES}"
                )
        
        # Make batch prediction
        prediction_start = time.time()
        predictions = await inference_service.predict_batch(
            batches=request.batches,
            model_version=request.model_version
        )
        inference_duration = time.time() - prediction_start
        
        # Record metrics
        batch_size = str(len(request.batches))
        metrics.PREDICTION_COUNT.labels(
            backend=inference_service.backend,
            endpoint_type='batch'
        ).inc(len(request.batches))
        
        metrics.INFERENCE_LATENCY.labels(
            backend=inference_service.backend,
            batch_size=batch_size
        ).observe(inference_duration)
        
        total_duration = time.time() - start_time
        
        logger.info(
            "Batch prediction completed",
            batch_size=len(request.batches),
            inference_duration_ms=inference_duration * 1000,
            total_duration_ms=total_duration * 1000,
            backend=inference_service.backend
        )
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(request.batches),
            model_version=inference_service.version,
            backend=inference_service.backend,
            inference_duration_ms=inference_duration * 1000,
            total_duration_ms=total_duration * 1000
        )
        
    except Exception as e:
        metrics.ERROR_COUNT.labels(
            error_type=type(e).__name__,
            endpoint='predict_batch'
        ).inc()
        
        logger.error("Batch prediction failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/async", response_model=AsyncTaskResponse)
async def predict_async(
    request: AsyncPredictionRequest,
    settings = Depends(get_settings)
) -> AsyncTaskResponse:
    """
    Asynchronous prediction endpoint.
    
    Submit a prediction job and get a task ID to check status later.
    
    - **inputs**: Feature vector or batch of feature vectors
    - **model_version**: Optional model version
    """
    try:
        # Validate input
        if isinstance(request.inputs[0], list):
            # Batch request
            if len(request.inputs) > settings.MAX_BATCH_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch too large. Maximum size: {settings.MAX_BATCH_SIZE}"
                )
            input_type = "batch"
        else:
            # Single request
            if len(request.inputs) > settings.MAX_FEATURES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Too many features. Maximum: {settings.MAX_FEATURES}"
                )
            input_type = "single"
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Submit async task
        task = async_predict_task.apply_async(
            args=[request.inputs, request.model_version, input_type],
            task_id=task_id
        )
        
        # Record metrics
        metrics.ASYNC_TASKS.labels(
            status='submitted',
            task_type=input_type
        ).inc()
        
        logger.info(
            "Async prediction task submitted",
            task_id=task_id,
            input_type=input_type,
            input_size=len(request.inputs)
        )
        
        return AsyncTaskResponse(
            task_id=task_id,
            status="submitted",
            estimated_duration_seconds=1.0 if input_type == "single" else len(request.inputs) * 0.1
        )
        
    except Exception as e:
        metrics.ERROR_COUNT.labels(
            error_type=type(e).__name__,
            endpoint='predict_async'
        ).inc()
        
        logger.error("Async prediction submission failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict/async/{task_id}", response_model=AsyncTaskStatusResponse)
async def get_async_prediction_status(task_id: str) -> AsyncTaskStatusResponse:
    """
    Get the status and result of an async prediction task.
    
    - **task_id**: The task ID returned from the async prediction endpoint
    """
    try:
        # Get task result
        result = celery_app.AsyncResult(task_id)
        
        if result.state == 'PENDING':
            response = AsyncTaskStatusResponse(
                task_id=task_id,
                status='pending',
                progress=0
            )
        elif result.state == 'STARTED':
            response = AsyncTaskStatusResponse(
                task_id=task_id,
                status='started',
                progress=50
            )
        elif result.state == 'SUCCESS':
            response = AsyncTaskStatusResponse(
                task_id=task_id,
                status='completed',
                progress=100,
                result=result.result,
                completed_at=time.time()
            )
        elif result.state == 'FAILURE':
            response = AsyncTaskStatusResponse(
                task_id=task_id,
                status='failed',
                progress=100,
                error=str(result.info),
                completed_at=time.time()
            )
        else:
            response = AsyncTaskStatusResponse(
                task_id=task_id,
                status=result.state.lower(),
                progress=75
            )
        
        logger.debug("Task status retrieved", task_id=task_id, status=result.state)
        
        return response
        
    except Exception as e:
        logger.error("Failed to get task status", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/predict/async/{task_id}")
async def cancel_async_prediction(task_id: str) -> Dict[str, Any]:
    """
    Cancel an async prediction task.
    
    - **task_id**: The task ID to cancel
    """
    try:
        celery_app.control.revoke(task_id, terminate=True)
        
        logger.info("Task cancelled", task_id=task_id)
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task cancellation requested"
        }
        
    except Exception as e:
        logger.error("Failed to cancel task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))