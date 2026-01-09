"""
FastAPI Application Entry Point
KK-OCR v2 Main Application
"""

import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.api.endpoints import router
from src.api.batch_endpoints import router as batch_router
from src.api.middleware import AuthenticationMiddleware, LoggingMiddleware
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.metrics import metrics_manager

# Initialize logger
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("Starting KK-OCR v2 Application", extra={
        "version": "2.1.0",
        "environment": settings.ENVIRONMENT
    })
    
    # Pre-load models (optional - can be lazy loaded)
    try:
        from src.modules.detector import YOLODetector
        from src.modules.extractor import VLMExtractor
        
        logger.info(f"Pre-loading models (pipeline mode: {settings.PIPELINE_MODE})...")
        
        # Store in app state for reuse
        # Only load YOLO if not using VLM-only mode
        if settings.PIPELINE_MODE != "vlm_only":
            app.state.detector = YOLODetector()
            logger.info("YOLO detector loaded")
        
        # Only load enhancer if using full pipeline mode
        if settings.PIPELINE_MODE == "full" and not settings.SKIP_ENHANCEMENT:
            from src.modules.enhancer_pretrained import PretrainedUNetEnhancer
            app.state.enhancer = PretrainedUNetEnhancer(
                model_name=settings.ENHANCEMENT_MODEL,
                encoder_name=settings.ENHANCEMENT_ENCODER,
                encoder_weights="imagenet",
                device=settings.DEVICE
            )
            logger.info("Pretrained U-Net enhancer loaded")
        else:
            logger.info("Skipping U-Net enhancer (SKIP_ENHANCEMENT=True or yolo_vlm mode)")
        
        app.state.extractor = VLMExtractor()
        logger.info(f"Models loaded successfully (mode: {settings.PIPELINE_MODE})")
    except Exception as e:
        logger.error(f"Failed to pre-load models: {str(e)}")
        # Continue anyway - will load on first request
    
    yield
    
    # Shutdown
    logger.info("Shutting down KK-OCR v2 Application")


# Initialize FastAPI app
app = FastAPI(
    title="KK-OCR v2 API",
    description="Indonesian Family Card OCR Pipeline - YOLO + U-Net + VLM",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middlewares
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthenticationMiddleware)

# Mount static files for UI
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Static files mounted from: {static_dir}")


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Record metrics
    if settings.ENABLE_METRICS:
        metrics_manager.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration=process_time
        )
    
    return response


# Include API routes
app.include_router(router, prefix="/v2")
app.include_router(batch_router, prefix="/v2")


# Root endpoint - serve HTML UI
@app.get("/", tags=["UI"])
async def root():
    """
    Serve the web UI
    """
    from fastapi.responses import FileResponse
    static_dir = Path(__file__).parent.parent.parent / "static"
    index_file = static_dir / "index.html"
    
    if index_file.exists():
        return FileResponse(index_file)
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "UI not found. Please ensure static files are present."}
        )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint
    Returns the status of the service
    """
    return {
        "status": "healthy",
        "version": "2.1.0",
        "service": "kk-ocr-v2"
    }


# Readiness check endpoint
@app.get("/ready", tags=["Health"])
async def readiness_check() -> Dict[str, bool | str]:
    """
    Readiness check endpoint
    Verifies that all dependencies are available
    """
    try:
        # Check if models are loaded
        if hasattr(app.state, 'detector'):
            return {
                "status": "ready",
                "models_loaded": True
            }
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "models_loaded": False
                }
            )
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(e)}
        )


# Metrics endpoint (Prometheus)
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus format
    """
    if not settings.ENABLE_METRICS:
        return JSONResponse(
            status_code=404,
            content={"error": "Metrics are disabled"}
        )
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler
    Catches all unhandled exceptions
    """
    logger.error(
        "Unhandled exception",
        extra={
            "path": request.url.path,
            "method": request.method,
            "error": str(exc),
            "error_type": type(exc).__name__
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )
