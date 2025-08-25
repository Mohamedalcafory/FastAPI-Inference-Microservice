"""
Response schemas for the FastAPI inference service.
"""
from typing import List, Optional, Union, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    prediction: Union[float, List[float]] = Field(..., description="Model prediction(s)")
    model_version: str = Field(..., description="Model version used")
    backend: str = Field(..., description="Model backend used")
    inference_duration_ms: float = Field(..., description="Inference duration in milliseconds")
    total_duration_ms: float = Field(..., description="Total request duration in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""
    predictions: List[Union[float, List[float]]] = Field(..., description="Model predictions")
    batch_size: int = Field(..., description="Number of predictions")
    model_version: str = Field(..., description="Model version used")
    backend: str = Field(..., description="Model backend used")
    inference_duration_ms: float = Field(..., description="Inference duration in milliseconds")
    total_duration_ms: float = Field(..., description="Total request duration in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")


class AsyncTaskResponse(BaseModel):
    """Response schema for async task submission."""
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    estimated_duration_seconds: Optional[float] = Field(None, description="Estimated task duration")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Task creation timestamp")


class AsyncTaskStatusResponse(BaseModel):
    """Response schema for async task status."""
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    progress: Optional[float] = Field(None, description="Task progress (0-100)")
    result: Optional[Union[float, List[float], List[Union[float, List[float]]]]] = Field(
        None, description="Task result (if completed)"
    )
    error: Optional[str] = Field(None, description="Error message (if failed)")
    created_at: datetime = Field(..., description="Task creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    model_version: Optional[str] = Field(None, description="Model version used")
    backend: Optional[str] = Field(None, description="Model backend used")


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    backend: str = Field(..., description="Model backend")
    version: str = Field(..., description="Model version")
    load_time: float = Field(..., description="Model load time in seconds")
    warmup_completed: bool = Field(..., description="Whether warmup is completed")
    feature_names: Optional[List[str]] = Field(None, description="Model feature names")
    n_features: Optional[int] = Field(None, description="Number of features")
    model_type: str = Field(..., description="Model type")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Info timestamp")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Service status")
    timestamp: float = Field(..., description="Unix timestamp")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")


class ReadyResponse(BaseModel):
    """Response schema for readiness check."""
    status: str = Field(..., description="Service readiness status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_backend: Optional[str] = Field(None, description="Model backend")
    model_version: Optional[str] = Field(None, description="Model version")
    timestamp: float = Field(..., description="Unix timestamp")


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    request_id: Optional[str] = Field(None, description="Request ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
