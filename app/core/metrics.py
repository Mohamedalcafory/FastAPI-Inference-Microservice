"""
Prometheus metrics configuration for the FastAPI inference service.
"""
from prometheus_client import Counter, Histogram, Gauge, Info


class Metrics:
    """Prometheus metrics for the inference service."""
    
    def __init__(self):
        # Request metrics
        self.REQUEST_COUNT = Counter(
            "inference_requests_total",
            "Total number of inference requests",
            ["endpoint", "method", "status"]
        )
        
        self.REQUEST_DURATION = Histogram(
            "inference_request_duration_seconds",
            "Request duration in seconds",
            ["endpoint", "method"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        )
        
        # Model metrics
        self.MODEL_PREDICTIONS = Counter(
            "model_predictions_total",
            "Total number of model predictions",
            ["backend", "endpoint_type"]
        )
        
        self.MODEL_INFERENCE_DURATION = Histogram(
            "model_inference_duration_seconds",
            "Model inference duration in seconds",
            ["backend", "batch_size"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.MODEL_LOAD_TIME = Histogram(
            "model_load_duration_seconds",
            "Model loading duration in seconds",
            ["backend"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        )
        
        # Async task metrics
        self.ASYNC_TASKS = Counter(
            "async_tasks_total",
            "Total number of async tasks",
            ["status", "task_type"]
        )
        
        self.ASYNC_TASK_DURATION = Histogram(
            "async_task_duration_seconds",
            "Async task duration in seconds",
            ["task_type", "status"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        )
        
        self.CELERY_QUEUE_DEPTH = Gauge(
            "celery_queue_depth",
            "Number of tasks in Celery queue",
            ["queue_name"]
        )
        
        # Error metrics
        self.ERRORS = Counter(
            "errors_total",
            "Total number of errors",
            ["error_type", "endpoint"]
        )
        
        # Model info
        self.MODEL_INFO = Info(
            "model_info",
            "Model information"
        )
        
        # Active connections (optional)
        self.ACTIVE_CONNECTIONS = Gauge(
            "active_connections",
            "Number of active connections"
        )


# Global metrics instance
metrics = Metrics()
