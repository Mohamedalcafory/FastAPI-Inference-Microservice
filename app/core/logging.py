"""
Structured logging configuration for the FastAPI inference service.
"""
import sys
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory


def configure_logging() -> None:
    """Configure structured logging."""
    settings = get_settings()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    import logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
    )


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_request(request_id: str, method: str, path: str, **kwargs: Any) -> None:
    """Log HTTP request."""
    logger = get_logger("http.request")
    logger.info(
        "HTTP request",
        request_id=request_id,
        method=method,
        path=path,
        **kwargs
    )


def log_response(request_id: str, status_code: int, duration_ms: float, **kwargs: Any) -> None:
    """Log HTTP response."""
    logger = get_logger("http.response")
    logger.info(
        "HTTP response",
        request_id=request_id,
        status_code=status_code,
        duration_ms=duration_ms,
        **kwargs
    )


def log_inference(
    request_id: str,
    backend: str,
    batch_size: int,
    duration_ms: float,
    **kwargs: Any
) -> None:
    """Log inference request."""
    logger = get_logger("inference")
    logger.info(
        "Inference completed",
        request_id=request_id,
        backend=backend,
        batch_size=batch_size,
        duration_ms=duration_ms,
        **kwargs
    )


def log_error(request_id: str, error: str, **kwargs: Any) -> None:
    """Log error."""
    logger = get_logger("error")
    logger.error(
        "Error occurred",
        request_id=request_id,
        error=error,
        **kwargs
    )


# Import at the end to avoid circular imports
from app.core.config import get_settings
