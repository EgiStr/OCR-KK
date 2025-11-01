"""
YOLO Detector Debug Script
Test and visualize YOLO detection results
"""

import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.modules.detector import YOLODetector, KK_FIELD_CLASSES
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def draw_detections(image: Image.Image, detections, save_path: str = "output_yolo_debug.jpg"):
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: PIL Image
        detections: List of Detection objects
        save_path: Path to save annotated image
    """
    # Create copy for drawing
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Color palette for different classes
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
        "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B739", "#52B788",
        "#FF8FA3", "#6C5CE7", "#A29BFE", "#FD79A8", "#FDCB6E",
        "#E17055", "#74B9FF", "#A29BFE", "#DFE6E9", "#00B894",
        "#00CEC9", "#0984E3"
    ]
    
    print(f"\n📦 Drawing {len(detections)} detections...")
    
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        color = colors[det.class_name.__hash__() % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Prepare label
        label = f"{det.class_name} ({det.confidence:.2f})"
        
        # Draw label background
        bbox = draw.textbbox((x1, y1 - 25), label, font=font_small)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 25), label, fill="white", font=font_small)
        
        print(f"  [{idx+1}] {det.class_name:20s} | Conf: {det.confidence:.3f} | BBox: [{x1:4d}, {y1:4d}, {x2:4d}, {y2:4d}] | Size: {x2-x1}x{y2-y1}")
    
    # Save annotated image
    img_draw.save(save_path)
    print(f"\n💾 Saved annotated image: {save_path}")
    
    return img_draw


def save_crops(detections, output_dir: str = "output_crops_yolo"):
    """
    Save individual crop images
    
    Args:
        detections: List of Detection objects
        output_dir: Directory to save crops
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n📁 Saving crops to: {output_dir}/")
    
    for idx, det in enumerate(detections):
        if det.crop is not None:
            crop_filename = f"{idx:02d}_{det.class_name}_{det.confidence:.2f}.jpg"
            crop_path = output_path / crop_filename
            det.crop.save(crop_path)
            print(f"  ✓ {crop_filename} ({det.crop.size[0]}x{det.crop.size[1]})")
    
    print(f"\n✅ Saved {len(detections)} crops")


def test_yolo_detector(image_path: str):
    """
    Test YOLO detector with detailed debugging
    
    Args:
        image_path: Path to test image
    """
    print("="*80)
    print("🔍 YOLO DETECTOR DEBUG TEST")
    print("="*80)
    
    # Check if model exists
    if not Path(settings.MODEL_PATH_YOLO).exists():
        print(f"❌ ERROR: YOLO model not found at: {settings.MODEL_PATH_YOLO}")
        print("   Please download or train the model first.")
        sys.exit(1)
    
    print(f"\n📋 Configuration:")
    print(f"   Model Path: {settings.MODEL_PATH_YOLO}")
    print(f"   Device: {settings.get_yolo_device()}")
    print(f"   Confidence Threshold: {settings.YOLO_CONFIDENCE_THRESHOLD}")
    print(f"   Input Size: {settings.YOLO_INPUT_SIZE}")
    print(f"   Number of Classes: {len(KK_FIELD_CLASSES)}")
    
    # Load image
    print(f"\n📸 Loading image: {image_path}")
    try:
        image = Image.open(image_path)
        print(f"   ✓ Image loaded: {image.size[0]}x{image.size[1]} ({image.mode})")
    except Exception as e:
        print(f"   ❌ Failed to load image: {e}")
        sys.exit(1)
    
    # Initialize detector
    print(f"\n🚀 Initializing YOLO Detector...")
    try:
        start_time = time.time()
        detector = YOLODetector()
        init_time = (time.time() - start_time) * 1000
        print(f"   ✓ Detector initialized in {init_time:.2f}ms")
        print(f"   Model device: {detector.device}")
    except Exception as e:
        print(f"   ❌ Failed to initialize detector: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run detection
    print(f"\n🎯 Running detection...")
    try:
        start_time = time.time()
        detections = detector.detect(image)
        detect_time = (time.time() - start_time) * 1000
        print(f"   ✓ Detection completed in {detect_time:.2f}ms")
        print(f"   Found {len(detections)} objects")
    except Exception as e:
        print(f"   ❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Analyze detections
    print(f"\n📊 Detection Analysis:")
    print(f"   Total detections: {len(detections)}")
    
    if len(detections) == 0:
        print("   ⚠️  No objects detected!")
        print("   Possible reasons:")
        print("   - Image quality too low")
        print("   - Confidence threshold too high")
        print("   - Image doesn't contain KK document")
        return
    
    # Group by class
    class_counts = {}
    confidence_by_class = {}
    for det in detections:
        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        if det.class_name not in confidence_by_class:
            confidence_by_class[det.class_name] = []
        confidence_by_class[det.class_name].append(det.confidence)
    
    print(f"\n   Detected classes ({len(class_counts)}):")
    for class_name, count in sorted(class_counts.items()):
        avg_conf = np.mean(confidence_by_class[class_name])
        print(f"   - {class_name:20s}: {count:2d} detection(s) | Avg Conf: {avg_conf:.3f}")
    
    # Confidence statistics
    all_confidences = [det.confidence for det in detections]
    print(f"\n   Confidence Statistics:")
    print(f"   - Min:  {min(all_confidences):.3f}")
    print(f"   - Max:  {max(all_confidences):.3f}")
    print(f"   - Mean: {np.mean(all_confidences):.3f}")
    print(f"   - Std:  {np.std(all_confidences):.3f}")
    
    # Size statistics
    sizes = [(det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1]) for det in detections]
    print(f"\n   Bounding Box Size Statistics:")
    print(f"   - Min area:  {min(sizes):,} pixels")
    print(f"   - Max area:  {max(sizes):,} pixels")
    print(f"   - Mean area: {int(np.mean(sizes)):,} pixels")
    
    # Draw detections
    print(f"\n🎨 Visualizing detections...")
    annotated_image = draw_detections(image, detections)
    
    # Save crops
    save_crops(detections)
    
    # Summary
    print("\n" + "="*80)
    print("✅ YOLO DEBUG TEST COMPLETED")
    print("="*80)
    print(f"\n📈 Performance Summary:")
    print(f"   Initialization: {init_time:.2f}ms")
    print(f"   Detection:      {detect_time:.2f}ms")
    print(f"   Total:          {init_time + detect_time:.2f}ms")
    print(f"\n📂 Output Files:")
    print(f"   - Annotated image: output_yolo_debug.jpg")
    print(f"   - Cropped fields:  output_crops_yolo/")
    print(f"\n💡 Next Steps:")
    print(f"   1. Check annotated image for detection accuracy")
    print(f"   2. Review cropped images for quality")
    print(f"   3. Adjust confidence threshold if needed (current: {settings.YOLO_CONFIDENCE_THRESHOLD})")
    

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_yolo_debug.py <image_path>")
        print("\nExample:")
        print("  python test_yolo_debug.py data/raw/sample_kk.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    test_yolo_detector(image_path)


if __name__ == "__main__":
    main()
