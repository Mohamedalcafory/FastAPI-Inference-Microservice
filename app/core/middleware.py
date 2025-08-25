"""
Middleware for the FastAPI inference service.
"""
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.logging import log_request, log_response, get_logger
from app.core.metrics import metrics


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect Prometheus metrics."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        endpoint = request.url.path
        method = request.method
        status = response.status_code
        
        metrics.REQUEST_COUNT.labels(
            endpoint=endpoint,
            method=method,
            status=status
        ).inc()
        
        metrics.REQUEST_DURATION.labels(
            endpoint=endpoint,
            method=method
        ).observe(duration)
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log requests and responses."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger("middleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request
        log_request(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            query_params=str(request.query_params),
            client_ip=request.client.host if request.client else None
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            log_response(
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=duration * 1000
            )
            
            return response
            
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            self.logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                duration_ms=duration * 1000
            )
            
            # Record error metric
            metrics.ERRORS.labels(
                error_type=type(e).__name__,
                endpoint=request.url.path
            ).inc()
            
            raise
