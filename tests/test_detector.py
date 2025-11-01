"""
Test YOLO Detector Module
"""

import pytest
import os
from PIL import Image
import numpy as np

from src.modules.detector import YOLODetector, Detection
from src.utils.config import get_settings

settings = get_settings()


@pytest.fixture
def dummy_image():
    """Create dummy image for testing"""
    # Create a simple RGB image
    img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def test_detection_class():
    """Test Detection class"""
    dummy_crop = Image.new("RGB", (100, 100), color="white")
    detection = Detection(
        bbox=[10, 20, 110, 120],
        class_name="no_kk",
        confidence=0.95,
        crop=dummy_crop
    )
    
    assert detection.bbox == [10, 20, 110, 120]
    assert detection.class_name == "no_kk"
    assert detection.confidence == 0.95
    
    # Test to_dict
    d = detection.to_dict()
    assert d["class_name"] == "no_kk"
    assert d["confidence"] == 0.95


# Note: Full YOLODetector tests require model file
# Add integration tests once model is available
@pytest.mark.skipif(
    not os.path.exists(settings.MODEL_PATH_YOLO),
    reason="Requires YOLO model file"
)
def test_yolo_detector_init():
    """Test YOLO detector initialization"""
    detector = YOLODetector()
    assert detector.model is not None


@pytest.mark.skipif(
    not os.path.exists(settings.MODEL_PATH_YOLO),
    reason="Requires YOLO model file"
)
def test_yolo_detection(dummy_image):
    """Test YOLO detection"""
    detector = YOLODetector()
    detections = detector.detect(dummy_image)
    assert isinstance(detections, list)
