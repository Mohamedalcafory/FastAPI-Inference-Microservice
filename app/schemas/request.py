"""
Request schemas for the FastAPI inference service.
"""
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    inputs: List[float] = Field(..., description="Input features for prediction")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    
    @validator('inputs')
    def validate_inputs(cls, v):
        if not v:
            raise ValueError("Inputs cannot be empty")
        if len(v) < 1:
            raise ValueError("At least one feature is required")
        return v


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""
    batches: List[List[float]] = Field(..., description="List of input feature batches")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    
    @validator('batches')
    def validate_batches(cls, v):
        if not v:
            raise ValueError("Batches cannot be empty")
        
        # Check batch size limit
        settings = get_settings()
        if len(v) > settings.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(v)} exceeds maximum of {settings.MAX_BATCH_SIZE}")
        
        # Validate each batch
        for i, batch in enumerate(v):
            if not batch:
                raise ValueError(f"Batch {i} cannot be empty")
            if len(batch) < 1:
                raise ValueError(f"Batch {i} must have at least one feature")
        
        return v


class AsyncPredictionRequest(BaseModel):
    """Request schema for async prediction."""
    inputs: Union[List[float], List[List[float]]] = Field(
        ..., 
        description="Input features (single or batch)"
    )
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    
    @validator('inputs')
    def validate_inputs(cls, v):
        if not v:
            raise ValueError("Inputs cannot be empty")
        
        # Handle single prediction
        if isinstance(v[0], (int, float)):
            if len(v) < 1:
                raise ValueError("At least one feature is required")
            return v
        
        # Handle batch prediction
        if isinstance(v[0], list):
            settings = get_settings()
            if len(v) > settings.MAX_BATCH_SIZE:
                raise ValueError(f"Batch size {len(v)} exceeds maximum of {settings.MAX_BATCH_SIZE}")
            
            for i, batch in enumerate(v):
                if not batch:
                    raise ValueError(f"Batch {i} cannot be empty")
                if len(batch) < 1:
                    raise ValueError(f"Batch {i} must have at least one feature")
            
            return v
        
        raise ValueError("Inputs must be a list of numbers or list of lists")


# Import at the end to avoid circular imports
from app.core.config import get_settings
