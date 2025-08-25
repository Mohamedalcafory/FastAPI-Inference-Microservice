"""
Model registry and loader for the FastAPI inference service.
"""
import time
import os
from typing import Any, List, Optional, Union
import numpy as np

import joblib
import onnxruntime as ort
from sklearn.base import BaseEstimator

from app.core.logging import get_logger
from app.core.metrics import metrics


class ModelRegistry:
    """Model registry for managing ML models."""
    
    def __init__(self):
        self.model: Optional[Union[BaseEstimator, ort.InferenceSession]] = None
        self.backend: str = "sklearn"
        self.version: str = "unknown"
        self.load_time: float = 0.0
        self.warmup_completed: bool = False
        self.logger = get_logger("model_registry")
    
    async def load_model(
        self,
        backend: str = "sklearn",
        model_path: str = "models/model.joblib",
        onnx_path: str = "models/model.onnx",
        warmup_enabled: bool = True,
        warmup_requests: int = 10
    ) -> None:
        """Load model from file."""
        start_time = time.time()
        
        try:
            if backend == "sklearn":
                await self._load_sklearn_model(model_path)
            elif backend == "onnx":
                await self._load_onnx_model(onnx_path)
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            
            self.backend = backend
            self.load_time = time.time() - start_time
            
            # Record load time metric
            metrics.MODEL_LOAD_TIME.labels(backend=backend).observe(self.load_time)
            
            self.logger.info(
                "Model loaded successfully",
                backend=backend,
                version=self.version,
                load_time=self.load_time
            )
            
            # Perform warmup if enabled
            if warmup_enabled:
                await self._warmup_model(warmup_requests)
                
        except Exception as e:
            self.logger.error(
                "Failed to load model",
                backend=backend,
                error=str(e)
            )
            raise
    
    async def _load_sklearn_model(self, model_path: str) -> None:
        """Load sklearn model from joblib file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        self.version = getattr(self.model, "version", "unknown")
        
        # Validate model
        if not hasattr(self.model, "predict"):
            raise ValueError("Model must have a predict method")
    
    async def _load_onnx_model(self, onnx_path: str) -> None:
        """Load ONNX model."""
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")
        
        # Create ONNX inference session
        self.model = ort.InferenceSession(onnx_path)
        self.version = "onnx_v1"  # Could be extracted from model metadata
        
        # Get input/output info
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
    
    async def _warmup_model(self, num_requests: int) -> None:
        """Warm up the model with dummy requests."""
        self.logger.info("Starting model warmup", num_requests=num_requests)
        
        try:
            # Generate dummy data based on model type
            if self.backend == "sklearn":
                # For sklearn, we need to know feature count
                if hasattr(self.model, "n_features_in_"):
                    n_features = self.model.n_features_in_
                else:
                    # Default to 4 features (common for iris-like datasets)
                    n_features = 4
                
                dummy_data = np.random.random((num_requests, n_features))
                
                for i in range(num_requests):
                    self.model.predict(dummy_data[i:i+1])
                    
            elif self.backend == "onnx":
                # For ONNX, we need to know input shape
                input_shape = self.model.get_inputs()[0].shape
                if len(input_shape) == 2:
                    # Batch dimension
                    n_features = input_shape[1]
                    dummy_data = np.random.random((1, n_features)).astype(np.float32)
                else:
                    # Single sample
                    n_features = input_shape[0]
                    dummy_data = np.random.random((1, n_features)).astype(np.float32)
                
                for i in range(num_requests):
                    self.model.run([self.output_name], {self.input_name: dummy_data})
            
            self.warmup_completed = True
            self.logger.info("Model warmup completed successfully")
            
        except Exception as e:
            self.logger.warning(
                "Model warmup failed, continuing without warmup",
                error=str(e)
            )
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def predict(self, inputs: Union[List[float], List[List[float]]]) -> Union[float, List[float], List[List[float]]]:
        """Make prediction using loaded model."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            if self.backend == "sklearn":
                return self._predict_sklearn(inputs)
            elif self.backend == "onnx":
                return self._predict_onnx(inputs)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
                
        except Exception as e:
            self.logger.error(
                "Prediction failed",
                backend=self.backend,
                error=str(e)
            )
            raise
    
    def _predict_sklearn(self, inputs: Union[List[float], List[List[float]]]) -> Union[float, List[float], List[List[float]]]:
        """Make prediction using sklearn model."""
        # Convert to numpy array
        if isinstance(inputs[0], (int, float)):
            # Single prediction
            X = np.array([inputs])
        else:
            # Batch prediction
            X = np.array(inputs)
        
        # Make prediction
        predictions = self.model.predict(X)
        
        # Return appropriate format
        if len(predictions) == 1:
            return float(predictions[0])
        else:
            return predictions.tolist()
    
    def _predict_onnx(self, inputs: Union[List[float], List[List[float]]]) -> Union[float, List[float], List[List[float]]]:
        """Make prediction using ONNX model."""
        # Convert to numpy array
        if isinstance(inputs[0], (int, float)):
            # Single prediction
            X = np.array([inputs], dtype=np.float32)
        else:
            # Batch prediction
            X = np.array(inputs, dtype=np.float32)
        
        # Make prediction
        predictions = self.model.run([self.output_name], {self.input_name: X})
        predictions = predictions[0]  # ONNX returns list of outputs
        
        # Return appropriate format
        if len(predictions) == 1:
            return float(predictions[0])
        else:
            return predictions.tolist()
    
    def get_model_info(self) -> dict:
        """Get model information."""
        if not self.is_loaded():
            return {"error": "Model not loaded"}
        
        info = {
            "backend": self.backend,
            "version": self.version,
            "load_time": self.load_time,
            "warmup_completed": self.warmup_completed,
            "model_type": type(self.model).__name__
        }
        
        # Add sklearn-specific info
        if self.backend == "sklearn" and hasattr(self.model, "feature_names_in_"):
            info["feature_names"] = self.model.feature_names_in_.tolist()
            info["n_features"] = self.model.n_features_in_
        
        # Add ONNX-specific info
        elif self.backend == "onnx":
            info["input_name"] = self.input_name
            info["output_name"] = self.output_name
        
        return info
