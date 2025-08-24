"""
Prometheus metrics for monitoring inference service.
"""
from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_client.metrics import MetricWrapperBase


class Metrics:
    """Centralized metrics collection."""
    
    def __init__(self):
        # Request metrics
        self.REQUEST_COUNT = Counter(
            'inference_requests_total',
            'Total number of inference requests',
            ['endpoint', 'method', 'status_code']
        )
        
        self.REQUEST_LATENCY = Histogram(
            'inference_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint', 'method'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
        )
        
        # Inference metrics
        self.INFERENCE_LATENCY = Histogram(
            'model_inference_duration_seconds',
            'Model inference duration in seconds',
            ['backend', 'batch_size'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
        )
        
        self.PREDICTION_COUNT = Counter(
            'model_predictions_total',
            'Total number of predictions made',
            ['backend', 'endpoint_type']
        )
        
        # Model metrics
        self.MODEL_LOAD_TIME = Histogram(
            'model_load_duration_seconds',
            'Time taken to load the model',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.MODEL_INFO = Info(
            'model_info',
            'Information about the loaded model'
        )
        
        self.COLD_START_COUNT = Counter(
            'model_cold_starts_total',
            'Number of cold starts',
            ['backend']
        )
        
        # Async task metrics
        self.ASYNC_TASKS = Counter(
            'async_tasks_total',
            'Total number of async tasks',
            ['status', 'task_type']
        )
        
        self.QUEUE_DEPTH = Gauge(
            'celery_queue_depth',
            'Number of tasks in queue'
        )
        
        self.TASK_DURATION = Histogram(
            'async_task_duration_seconds',
            'Async task duration in seconds',
            ['task_type', 'status'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)
        )
        
        # System metrics
        self.ACTIVE_CONNECTIONS = Gauge(
            'active_connections',
            'Number of active connections'
        )
        
        self.ERROR_COUNT = Counter(
            'errors_total',
            'Total number of errors',
            ['error_type', 'endpoint']
        )
        
        # Cache metrics (if implemented)
        self.CACHE_HITS = Counter(
            'cache_hits_total',
            'Number of cache hits',
            ['cache_type']
        )
        
        self.CACHE_MISSES = Counter(
            'cache_misses_total',
            'Number of cache misses',
            ['cache_type']
        )


# Global metrics instance
metrics = Metrics()