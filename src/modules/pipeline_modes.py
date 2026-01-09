"""
Pipeline Modes Module
Abstraction for different OCR pipeline configurations:
- VLM Only: Direct VLM extraction without detection
- YOLO + VLM: Detection then VLM extraction (recommended)
- Full: Detection + Enhancement + VLM extraction
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from PIL import Image
import time

from src.modules.detector import Detection, YOLODetector
from src.modules.extractor import VLMExtractor
from src.api.models import KKExtractionResponse
from src.utils.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class BasePipeline(ABC):
    """Abstract base class for OCR pipelines"""
    
    @abstractmethod
    async def process(
        self,
        image: Image.Image,
        filename: str
    ) -> Tuple[KKExtractionResponse, dict]:
        """
        Process image through the pipeline
        
        Args:
            image: PIL Image to process
            filename: Original filename
            
        Returns:
            Tuple of (extraction response, timing metrics)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Pipeline name for logging"""
        pass


class VLMOnlyPipeline(BasePipeline):
    """
    Direct VLM extraction without any detection or enhancement.
    
    Best for:
    - Clean, high-quality document scans
    - Fastest processing time
    - Lowest memory footprint
    """
    
    def __init__(self, extractor: Optional[VLMExtractor] = None):
        self.extractor = extractor or VLMExtractor()
        logger.info("Initialized VLMOnlyPipeline")
    
    @property
    def name(self) -> str:
        return "vlm_only"
    
    async def process(
        self,
        image: Image.Image,
        filename: str
    ) -> Tuple[KKExtractionResponse, dict]:
        """Process image directly with VLM"""
        metrics = {"pipeline": self.name}
        
        # Direct VLM extraction (no detection)
        extraction_start = time.time()
        result = await self.extractor.extract_direct(
            image=image,
            source_filename=filename
        )
        metrics["extraction_time"] = time.time() - extraction_start
        metrics["total_time"] = metrics["extraction_time"]
        
        return result, metrics


class YoloVLMPipeline(BasePipeline):
    """
    YOLO detection followed by VLM extraction (RECOMMENDED).
    
    Best for:
    - Balance of accuracy and speed
    - Varied document quality
    - Structural information needed for row association
    """
    
    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        extractor: Optional[VLMExtractor] = None
    ):
        self.detector = detector or YOLODetector()
        self.extractor = extractor or VLMExtractor()
        logger.info("Initialized YoloVLMPipeline")
    
    @property
    def name(self) -> str:
        return "yolo_vlm"
    
    async def process(
        self,
        image: Image.Image,
        filename: str
    ) -> Tuple[KKExtractionResponse, dict]:
        """Process with YOLO detection then VLM extraction"""
        metrics = {"pipeline": self.name}
        total_start = time.time()
        
        # Step 1: YOLO Detection
        detection_start = time.time()
        detections = self.detector.detect(image)
        metrics["detection_time"] = time.time() - detection_start
        metrics["num_detections"] = len(detections)
        
        logger.debug(
            f"YOLO detection completed",
            detections=len(detections),
            time_ms=int(metrics["detection_time"] * 1000)
        )
        
        # Step 2: VLM Extraction with detection context
        extraction_start = time.time()
        result = await self.extractor.extract_with_detections(
            image=image,
            detections=detections,
            source_filename=filename
        )
        metrics["extraction_time"] = time.time() - extraction_start
        
        metrics["total_time"] = time.time() - total_start
        
        return result, metrics


class FullPipeline(BasePipeline):
    """
    Full pipeline with YOLO detection, U-Net enhancement, and VLM extraction.
    
    Best for:
    - Low-quality or noisy document scans
    - Maximum accuracy requirements
    - When enhancement is proven to help
    """
    
    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        enhancer = None,
        extractor: Optional[VLMExtractor] = None
    ):
        self.detector = detector or YOLODetector()
        self.extractor = extractor or VLMExtractor()
        
        # Lazy load enhancer to avoid import if not needed
        if enhancer is None:
            from src.modules.enhancer_pretrained import PretrainedUNetEnhancer
            self.enhancer = PretrainedUNetEnhancer(
                encoder_name=settings.ENHANCEMENT_ENCODER,
                encoder_weights="imagenet",
                device=settings.DEVICE
            )
        else:
            self.enhancer = enhancer
            
        logger.info("Initialized FullPipeline")
    
    @property
    def name(self) -> str:
        return "full"
    
    async def process(
        self,
        image: Image.Image,
        filename: str
    ) -> Tuple[KKExtractionResponse, dict]:
        """Process with detection, enhancement, then VLM extraction"""
        metrics = {"pipeline": self.name}
        total_start = time.time()
        
        # Step 1: YOLO Detection
        detection_start = time.time()
        detections = self.detector.detect(image)
        metrics["detection_time"] = time.time() - detection_start
        metrics["num_detections"] = len(detections)
        
        # Step 2: Image Enhancement
        enhancement_start = time.time()
        
        from src.utils.validators import crop_with_bbox
        cropped_images = [
            crop_with_bbox(image, det.bbox) for det in detections
        ]
        
        enhancement_method = getattr(settings, 'ENHANCEMENT_METHOD', 'hybrid')
        enhanced_crops = self.enhancer.enhance_batch(
            images=cropped_images,
            detections=detections,
            method=enhancement_method
        )
        metrics["enhancement_time"] = time.time() - enhancement_start
        
        # Convert to extractor-compatible format
        from src.modules.enhancer import EnhancedCrop as ExtractorEnhancedCrop
        extractor_crops = []
        for enh_crop in enhanced_crops:
            det = enh_crop.detection
            bbox = [int(x) for x in det.bbox]
            extractor_crop = ExtractorEnhancedCrop(
                original=enh_crop.original_image,
                enhanced=enh_crop.enhanced_image,
                bbox=bbox,
                class_name=det.class_name
            )
            extractor_crops.append(extractor_crop)
        
        # Step 3: VLM Extraction
        extraction_start = time.time()
        result = await self.extractor.extract(
            original_image=image,
            detections=detections,
            enhanced_crops=extractor_crops,
            source_filename=filename
        )
        metrics["extraction_time"] = time.time() - extraction_start
        
        metrics["total_time"] = time.time() - total_start
        
        return result, metrics


def get_pipeline(mode: Optional[str] = None) -> BasePipeline:
    """
    Factory function to get pipeline based on mode.
    
    Args:
        mode: Pipeline mode ("vlm_only", "yolo_vlm", "full")
              Defaults to settings.PIPELINE_MODE
    
    Returns:
        Configured pipeline instance
    """
    mode = mode or settings.PIPELINE_MODE
    
    if mode == "vlm_only":
        return VLMOnlyPipeline()
    elif mode == "yolo_vlm":
        return YoloVLMPipeline()
    elif mode == "full":
        return FullPipeline()
    else:
        logger.warning(f"Unknown pipeline mode '{mode}', defaulting to yolo_vlm")
        return YoloVLMPipeline()
