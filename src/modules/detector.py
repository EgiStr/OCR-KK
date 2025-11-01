"""
YOLO Detector Module
Detects 22 field classes from KK document
"""

from typing import List, Dict, Any
from pathlib import Path

import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO

from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.validators import crop_with_bbox

settings = get_settings()
logger = get_logger(__name__)


# KK Field Classes (22 classes as per PRD)
KK_FIELD_CLASSES = [
    "no_kk", "kepala_keluarga", "alamat", "rt", "rw", 
    "desa_kelurahan", "kecamatan", "kabupaten_kota", "provinsi", "kode_pos",
    "nik", "nama_lengkap", "jenis_kelamin", "tempat_lahir", "tanggal_lahir",
    "agama", "pendidikan", "pekerjaan", "status_perkawinan", "status_keluarga",
    "kewarganegaraan", "tanggal_pembuatan"
]


class Detection:
    """
    Single detection result
    """
    def __init__(
        self,
        bbox: List[int],
        class_name: str,
        confidence: float,
        crop: Image.Image
    ):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_name = class_name
        self.confidence = confidence
        self.crop = crop
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "bbox": self.bbox,
            "class_name": self.class_name,
            "confidence": float(self.confidence)
        }


class YOLODetector:
    """
    YOLO-based field detector for KK documents
    """
    
    def __init__(self):
        """Initialize YOLO detector"""
        self.model_path = settings.MODEL_PATH_YOLO
        self.device = settings.get_yolo_device()
        self.confidence_threshold = settings.YOLO_CONFIDENCE_THRESHOLD
        self.input_size = settings.YOLO_INPUT_SIZE
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            logger.info(
                "Loading YOLO model",
                extra={
                    "model_path": Path(self.model_path).name,
                    "device": self.device
                }
            )
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"YOLO model not found at {self.model_path}. "
                    f"Please download or train the model first."
                )
            
            # Load model
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            logger.info(
                "YOLO model loaded successfully",
                extra={
                    "device": self.device,
                    "num_classes": len(self.model.names) if hasattr(self.model, 'names') else 'unknown'
                }
            )
            
            # Update metrics
            from src.utils.metrics import metrics_manager
            metrics_manager.set_model_loaded("yolo", True)
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
    
    def detect(self, image: Image.Image) -> List[Detection]:
        """
        Detect fields in KK document
        
        Args:
            image: Input PIL Image
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            raise RuntimeError("YOLO model not loaded")
        
        try:
            logger.debug(
                "Running YOLO detection",
                extra={"image_size": image.size}
            )
            
            # Run inference
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                imgsz=self.input_size,
                verbose=False,
                device=self.device
            )
            
            # Parse results
            detections = []
            
            if len(results) > 0:
                result = results[0]
                
                # Extract boxes
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    
                    # Get class and confidence
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    
                    # Get class name
                    class_name = self.model.names.get(class_id, f"class_{class_id}")
                    
                    # Crop image
                    crop = crop_with_bbox(image, bbox)
                    
                    # Create detection
                    detection = Detection(
                        bbox=bbox,
                        class_name=class_name,
                        confidence=confidence,
                        crop=crop
                    )
                    
                    detections.append(detection)
                    
                    logger.debug(
                        "Detection found",
                        extra={
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": bbox
                        }
                    )
            
            logger.info(
                "YOLO detection completed",
                extra={"num_detections": len(detections)}
            )
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"YOLO detection failed: {str(e)}")
    
    def detect_batch(self, images: List[Image.Image]) -> List[List[Detection]]:
        """
        Detect fields in multiple images (batch processing)
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of detection lists
        """
        return [self.detect(image) for image in images]
    
    def get_detections_by_class(
        self,
        detections: List[Detection],
        class_name: str
    ) -> List[Detection]:
        """
        Filter detections by class name
        
        Args:
            detections: List of detections
            class_name: Class name to filter
            
        Returns:
            Filtered detections
        """
        return [d for d in detections if d.class_name == class_name]
    
    def sort_detections_by_position(
        self,
        detections: List[Detection],
        by: str = "top_to_bottom"
    ) -> List[Detection]:
        """
        Sort detections by position
        
        Args:
            detections: List of detections
            by: Sort order ("top_to_bottom", "left_to_right", "row_wise")
            
        Returns:
            Sorted detections
        """
        if by == "top_to_bottom":
            return sorted(detections, key=lambda d: d.bbox[1])  # Sort by y1
        elif by == "left_to_right":
            return sorted(detections, key=lambda d: d.bbox[0])  # Sort by x1
        elif by == "row_wise":
            # Sort by row (y), then by column (x)
            return sorted(detections, key=lambda d: (d.bbox[1], d.bbox[0]))
        else:
            return detections
    
    def group_detections_by_row(
        self,
        detections: List[Detection],
        row_threshold: int = 20
    ) -> List[List[Detection]]:
        """
        Group detections by row (for family members)
        
        Args:
            detections: List of detections
            row_threshold: Maximum vertical distance to consider same row
            
        Returns:
            List of detection groups (rows)
        """
        if not detections:
            return []
        
        # Sort by y position
        sorted_dets = sorted(detections, key=lambda d: d.bbox[1])
        
        rows = []
        current_row = [sorted_dets[0]]
        current_y = sorted_dets[0].bbox[1]
        
        for detection in sorted_dets[1:]:
            y = detection.bbox[1]
            
            if abs(y - current_y) <= row_threshold:
                # Same row
                current_row.append(detection)
            else:
                # New row
                rows.append(current_row)
                current_row = [detection]
                current_y = y
        
        # Add last row
        if current_row:
            rows.append(current_row)
        
        return rows
