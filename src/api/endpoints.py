"""
API Endpoints
Route handlers for KK-OCR v2 extraction
"""

import time
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Depends
from fastapi.responses import JSONResponse

from src.api.models import KKExtractionResponse, ErrorResponse
from src.modules.detector import YOLODetector
from src.modules.enhancer_pretrained import PretrainedUNetEnhancer
from src.modules.extractor import VLMExtractor
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.validators import validate_file, load_image_from_upload
from src.utils.metrics import metrics_manager

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


# Dependency to get or create detector
async def get_detector(request: Request) -> YOLODetector:
    """Get YOLO detector instance"""
    if hasattr(request.app.state, 'detector'):
        return request.app.state.detector
    return YOLODetector()


# Dependency to get or create enhancer
async def get_enhancer(request: Request) -> PretrainedUNetEnhancer:
    """Get Pretrained U-Net enhancer instance"""
    if hasattr(request.app.state, 'enhancer'):
        return request.app.state.enhancer
    # Fallback: create new instance if not pre-loaded
    return PretrainedUNetEnhancer(
        encoder_name=settings.ENHANCEMENT_ENCODER,
        encoder_weights="imagenet",
        device=settings.DEVICE
    )


# Dependency to get or create extractor
async def get_extractor(request: Request) -> VLMExtractor:
    """Get VLM extractor instance"""
    if hasattr(request.app.state, 'extractor'):
        return request.app.state.extractor
    return VLMExtractor()


@router.post(
    "/extract/kk",
    response_model=KKExtractionResponse,
    responses={
        200: {"model": KKExtractionResponse, "description": "Successful extraction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    tags=["Extraction"],
    summary="Extract data from Kartu Keluarga",
    description="""
    Extract structured data from Indonesian Family Card (Kartu Keluarga) document.
    
    **Process:**
    1. YOLO detection of 22 field classes
    2. U-Net enhancement of detected regions
    3. VLM extraction using Gemini 1.5 Pro
    
    **Supported formats:** JPEG, PNG, PDF (first page only)
    
    **Max file size:** 10MB
    
    **Authentication:** Required (Bearer token)
    """
)
async def extract_kk(
    file: UploadFile = File(..., description="KK document image"),
    detector: YOLODetector = Depends(get_detector),
    enhancer: PretrainedUNetEnhancer = Depends(get_enhancer),
    extractor: VLMExtractor = Depends(get_extractor)
) -> KKExtractionResponse:
    """
    Extract structured data from Kartu Keluarga document
    
    Args:
        file: Uploaded KK document image
        detector: YOLO detector instance
        enhancer: U-Net enhancer instance
        extractor: VLM extractor instance
        
    Returns:
        KKExtractionResponse with metadata, header, family members, and footer
        
    Raises:
        HTTPException: On validation or processing errors
    """
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    logger.info(
        "Processing KK extraction request",
        extra={
            "request_id": request_id,
            "filename": file.filename,
            "content_type": file.content_type
        }
    )
    
    try:
        # 1. Validate file
        validate_file(file)
        
        # 2. Load image
        image = await load_image_from_upload(file)
        logger.debug(f"Image loaded: {image.size}")
        
        # 3. YOLO Detection
        detection_start = time.time()
        detections = detector.detect(image)
        detection_time = time.time() - detection_start
        
        logger.info(
            "YOLO detection completed",
            extra={
                "request_id": request_id,
                "num_detections": len(detections),
                "detection_time_ms": int(detection_time * 1000)
            }
        )
        
        if settings.ENABLE_METRICS:
            metrics_manager.record_detection_time(detection_time)
            metrics_manager.record_detections(len(detections))
        
        # 4. Image Enhancement (Pretrained U-Net)
        enhancement_start = time.time()
        
        # Crop images from detections
        from src.utils.validators import crop_with_bbox
        cropped_images = [
            crop_with_bbox(image, det.bbox) for det in detections
        ]
        
        # Enhance with pretrained U-Net (no training required!)
        enhancement_method = getattr(settings, 'ENHANCEMENT_METHOD', 'hybrid')
        enhanced_crops = enhancer.enhance_batch(
            images=cropped_images,
            detections=detections,
            method=enhancement_method  # hybrid, classical, or deep
        )
        enhancement_time = time.time() - enhancement_start
        
        logger.info(
            "Pretrained U-Net enhancement completed",
            extra={
                "request_id": request_id,
                "num_enhanced": len(enhanced_crops),
                "enhancement_time_ms": int(enhancement_time * 1000),
                "method": enhancement_method
            }
        )
        
        if settings.ENABLE_METRICS:
            metrics_manager.record_enhancement_time(enhancement_time)
        
        # Convert PretrainedUNetEnhancer EnhancedCrop to extractor-compatible format
        # extractor.py expects EnhancedCrop with bbox attribute
        from src.modules.enhancer import EnhancedCrop as ExtractorEnhancedCrop
        
        extractor_crops = []
        for enh_crop in enhanced_crops:
            # Get bbox from detection - it's already a list [x1, y1, x2, y2]
            det = enh_crop.detection
            bbox = [int(x) for x in det.bbox]
            
            # Create extractor-compatible EnhancedCrop
            extractor_crop = ExtractorEnhancedCrop(
                original=enh_crop.original_image,
                enhanced=enh_crop.enhanced_image,
                bbox=bbox,
                class_name=det.class_name
            )
            extractor_crops.append(extractor_crop)
        
        # 5. VLM Extraction
        extraction_start = time.time()
        result = await extractor.extract(
            original_image=image,
            detections=detections,
            enhanced_crops=extractor_crops,
            source_filename=file.filename
        )
        extraction_time = time.time() - extraction_start
        
        logger.info(
            "VLM extraction completed",
            extra={
                "request_id": request_id,
                "extraction_time_ms": int(extraction_time * 1000)
            }
        )
        
        if settings.ENABLE_METRICS:
            metrics_manager.record_extraction_time(extraction_time)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        logger.info(
            "KK extraction completed successfully",
            extra={
                "request_id": request_id,
                "total_time_ms": int(total_time * 1000),
                "detection_time_ms": int(detection_time * 1000),
                "enhancement_time_ms": int(enhancement_time * 1000),
                "extraction_time_ms": int(extraction_time * 1000)
            }
        )
        
        if settings.ENABLE_METRICS:
            metrics_manager.record_total_time(total_time)
            metrics_manager.increment_success_counter()
        
        return result
        
    except ValueError as e:
        # Validation errors
        logger.warning(
            "Validation error",
            extra={"request_id": request_id, "error": str(e)}
        )
        if settings.ENABLE_METRICS:
            metrics_manager.increment_error_counter("validation")
        raise HTTPException(status_code=400, detail=str(e))
        
    except TimeoutError as e:
        # Timeout errors
        logger.error(
            "Processing timeout",
            extra={"request_id": request_id, "error": str(e)}
        )
        if settings.ENABLE_METRICS:
            metrics_manager.increment_error_counter("timeout")
        raise HTTPException(status_code=504, detail="Processing timeout exceeded")
        
    except Exception as e:
        # Unexpected errors
        logger.error(
            "Processing error",
            extra={
                "request_id": request_id,
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        if settings.ENABLE_METRICS:
            metrics_manager.increment_error_counter("processing")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}" if settings.DEBUG else "Internal processing error"
        )


@router.get(
    "/info",
    tags=["Info"],
    summary="Get API information"
)
async def get_api_info():
    """
    Get API information and model versions
    """
    return {
        "api_version": "2.1.0",
        "pipeline_stages": ["YOLO Detection", "U-Net Enhancement", "VLM Extraction"],
        "supported_formats": ["JPEG", "PNG", "PDF"],
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "models": {
            "yolo": "v1.0.0 (mAP@0.5-0.95 = 0.886)",
            "unet": "v1.0.0",
            "vlm": settings.GEMINI_MODEL
        }
    }
