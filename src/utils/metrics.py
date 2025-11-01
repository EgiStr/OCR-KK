"""
Prometheus Metrics Manager
Track performance and usage metrics
"""

from prometheus_client import Counter, Histogram, Gauge, Summary
from typing import Optional

from src.utils.config import get_settings

settings = get_settings()


class MetricsManager:
    """
    Centralized metrics management for Prometheus
    """
    
    def __init__(self):
        # Request metrics
        self.request_counter = Counter(
            "kk_ocr_requests_total",
            "Total number of requests",
            ["endpoint", "method", "status"]
        )
        
        self.request_duration = Histogram(
            "kk_ocr_request_duration_seconds",
            "Request duration in seconds",
            ["endpoint"],
            buckets=(0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
        )
        
        # Pipeline stage metrics
        self.detection_duration = Histogram(
            "kk_ocr_detection_duration_seconds",
            "YOLO detection duration in seconds",
            buckets=(0.01, 0.05, 0.1, 0.15, 0.2, 0.5)
        )
        
        self.enhancement_duration = Histogram(
            "kk_ocr_enhancement_duration_seconds",
            "U-Net enhancement duration in seconds",
            buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
        )
        
        self.extraction_duration = Histogram(
            "kk_ocr_extraction_duration_seconds",
            "VLM extraction duration in seconds",
            buckets=(0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
        )
        
        self.total_duration = Histogram(
            "kk_ocr_total_duration_seconds",
            "Total end-to-end duration in seconds",
            buckets=(0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 15.0)
        )
        
        # Detection metrics
        self.detections_counter = Counter(
            "kk_ocr_detections_total",
            "Total number of field detections",
            ["class_name"]
        )
        
        self.detections_per_document = Histogram(
            "kk_ocr_detections_per_document",
            "Number of detections per document",
            buckets=(5, 10, 15, 20, 25, 30, 40, 50)
        )
        
        # Success/Error metrics
        self.success_counter = Counter(
            "kk_ocr_success_total",
            "Total number of successful extractions"
        )
        
        self.error_counter = Counter(
            "kk_ocr_errors_total",
            "Total number of errors",
            ["error_type"]
        )
        
        # Model metrics
        self.model_loaded = Gauge(
            "kk_ocr_model_loaded",
            "Whether model is loaded (1=loaded, 0=not loaded)",
            ["model_name"]
        )
        
        # File size metrics
        self.file_size = Histogram(
            "kk_ocr_file_size_bytes",
            "Input file size in bytes",
            buckets=(100000, 500000, 1000000, 2000000, 5000000, 10000000)
        )
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float
    ):
        """Record HTTP request metrics"""
        if not settings.ENABLE_METRICS:
            return
        
        status = f"{status_code // 100}xx"
        self.request_counter.labels(
            endpoint=endpoint,
            method=method,
            status=status
        ).inc()
        
        self.request_duration.labels(endpoint=endpoint).observe(duration)
    
    def record_detection_time(self, duration: float):
        """Record YOLO detection time"""
        if settings.ENABLE_METRICS:
            self.detection_duration.observe(duration)
    
    def record_enhancement_time(self, duration: float):
        """Record U-Net enhancement time"""
        if settings.ENABLE_METRICS:
            self.enhancement_duration.observe(duration)
    
    def record_extraction_time(self, duration: float):
        """Record VLM extraction time"""
        if settings.ENABLE_METRICS:
            self.extraction_duration.observe(duration)
    
    def record_total_time(self, duration: float):
        """Record total processing time"""
        if settings.ENABLE_METRICS:
            self.total_duration.observe(duration)
    
    def record_detections(self, count: int, class_name: Optional[str] = None):
        """Record number of detections"""
        if not settings.ENABLE_METRICS:
            return
        
        self.detections_per_document.observe(count)
        
        if class_name:
            self.detections_counter.labels(class_name=class_name).inc()
    
    def increment_success_counter(self):
        """Increment successful extraction counter"""
        if settings.ENABLE_METRICS:
            self.success_counter.inc()
    
    def increment_error_counter(self, error_type: str):
        """Increment error counter"""
        if settings.ENABLE_METRICS:
            self.error_counter.labels(error_type=error_type).inc()
    
    def set_model_loaded(self, model_name: str, loaded: bool):
        """Set model loaded status"""
        if settings.ENABLE_METRICS:
            self.model_loaded.labels(model_name=model_name).set(1 if loaded else 0)
    
    def record_file_size(self, size_bytes: int):
        """Record input file size"""
        if settings.ENABLE_METRICS:
            self.file_size.observe(size_bytes)


# Global metrics manager instance
metrics_manager = MetricsManager()
