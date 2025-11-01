"""
Logging Utilities
PII-safe structured logging
"""

import sys
import logging
import structlog
from typing import Any, Dict
from pathlib import Path

from src.utils.config import get_settings

settings = get_settings()


# ==================== PII Scrubbing ====================

PII_FIELDS = [
    "nik", "nama_lengkap", "nama", "name", "nama_ayah", "nama_ibu",
    "tempat_lahir", "no_kk", "alamat", "address", "no_paspor", "no_kitap"
]


def scrub_pii(logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove PII from log event
    
    This is a structlog processor that scrubs PII fields from logs.
    Signature: (logger, method_name, event_dict) -> event_dict
    """
    if not settings.ENABLE_PII_SCRUBBING:
        return event_dict
    
    # Scrub event message
    event = event_dict.get("event", "")
    if isinstance(event, str):
        for field in PII_FIELDS:
            if field.lower() in event.lower():
                event_dict["event"] = "[SCRUBBED - PII detected]"
                break
    
    # Scrub extra fields
    for key in list(event_dict.keys()):
        if key.lower() in PII_FIELDS:
            event_dict[key] = "[REDACTED]"
    
    return event_dict


# ==================== Logger Configuration ====================

def configure_logging():
    """
    Configure structured logging
    """
    # Set log level
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        scrub_pii,  # PII scrubbing processor
    ]
    
    # Add appropriate renderer based on format
    if settings.LOG_FORMAT == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


# Initialize logging on import
configure_logging()


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# ==================== Logging Utilities ====================

def log_model_load(logger: structlog.stdlib.BoundLogger, model_name: str, model_path: str):
    """
    Log model loading (without sensitive information)
    """
    logger.info(
        "Loading model",
        extra={
            "model_name": model_name,
            "model_path": Path(model_path).name  # Only filename, not full path
        }
    )


def log_performance_metrics(
    logger: structlog.stdlib.BoundLogger,
    operation: str,
    duration_ms: int,
    additional_metrics: Dict[str, Any] = None
):
    """
    Log performance metrics
    """
    metrics = {
        "operation": operation,
        "duration_ms": duration_ms
    }
    
    if additional_metrics:
        metrics.update(additional_metrics)
    
    logger.info("Performance metric", extra=metrics)


def log_error_without_pii(
    logger: structlog.stdlib.BoundLogger,
    error_message: str,
    error_type: str,
    context: Dict[str, Any] = None
):
    """
    Log error without exposing PII
    """
    log_data = {
        "error_message": error_message,
        "error_type": error_type
    }
    
    if context:
        # Filter out PII from context
        safe_context = {
            k: v for k, v in context.items()
            if k.lower() not in PII_FIELDS
        }
        log_data.update(safe_context)
    
    logger.error("Error occurred", extra=log_data)
