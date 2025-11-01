"""
Full Pipeline Debug Script
Test complete KK-OCR v2 pipeline with debugging
"""

import sys
import time
import json
import asyncio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from src.modules.detector import YOLODetector
from src.modules.enhancer_pretrained import PretrainedUNetEnhancer
from src.modules.extractor import VLMExtractor
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.validators import crop_with_bbox

logger = get_logger(__name__)
settings = get_settings()


def save_pipeline_visualization(
    original_image,
    detections,
    enhanced_crops,
    extraction_result,
    output_path="output_pipeline_visualization.jpg"
):
    """
    Create comprehensive pipeline visualization
    
    Args:
        original_image: Original KK image
        detections: List of Detection objects
        enhanced_crops: List of enhanced crop results
        extraction_result: Final extraction result
        output_path: Path to save visualization
    """
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Original image with detections
    ax1 = fig.add_subplot(gs[0, :2])
    img_draw = original_image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Draw bounding boxes
    for det in detections[:10]:  # Limit to 10 for visibility
        x1, y1, x2, y2 = det.bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1-20), det.class_name, fill="red")
    
    ax1.imshow(img_draw)
    ax1.set_title(f"1. YOLO Detection ({len(detections)} fields)", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Sample crops (before enhancement)
    ax2 = fig.add_subplot(gs[0, 2:])
    sample_crops = [det.crop for det in detections[:4]]
    if sample_crops:
        combined = Image.new('RGB', (sum(c.width for c in sample_crops), max(c.height for c in sample_crops)))
        x_offset = 0
        for crop in sample_crops:
            combined.paste(crop, (x_offset, 0))
            x_offset += crop.width
        ax2.imshow(combined)
    ax2.set_title("2. Sample Crops (Original)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. Sample crops (after enhancement)
    ax3 = fig.add_subplot(gs[1, :2])
    if enhanced_crops:
        sample_enhanced = [ec.enhanced_image for ec in enhanced_crops[:4]]
        if sample_enhanced:
            combined = Image.new('RGB' if sample_enhanced[0].mode == 'RGB' else 'L',
                               (sum(c.width for c in sample_enhanced), max(c.height for c in sample_enhanced)))
            x_offset = 0
            for crop in sample_enhanced:
                combined.paste(crop, (x_offset, 0))
                x_offset += crop.width
            ax3.imshow(combined, cmap='gray' if combined.mode == 'L' else None)
    ax3.set_title("3. Enhanced Crops", fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 4. Extraction summary
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.axis('off')
    
    summary_text = "4. VLM Extraction Result\n\n"
    if extraction_result:
        if "header" in extraction_result:
            header = extraction_result["header"]
            summary_text += f"Header:\n"
            summary_text += f"  No. KK: {header.get('no_kk', 'N/A')}\n"
            summary_text += f"  Kepala Keluarga: {header.get('kepala_keluarga', 'N/A')}\n"
            summary_text += f"  Alamat: {header.get('alamat', 'N/A')[:30]}...\n\n"
        
        if "anggota_keluarga" in extraction_result:
            members = extraction_result["anggota_keluarga"]
            summary_text += f"Family Members: {len(members)}\n"
            for i, m in enumerate(members[:3], 1):
                summary_text += f"  {i}. {m.get('nama_lengkap', 'N/A')}\n"
            if len(members) > 3:
                summary_text += f"  ... and {len(members)-3} more\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Performance metrics
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    plt.suptitle("KK-OCR v2 Pipeline Debug Visualization", fontsize=18, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Saved pipeline visualization: {output_path}")


async def test_full_pipeline(image_path: str, save_visualization: bool = True):
    """
    Test complete pipeline with debugging
    
    Args:
        image_path: Path to KK image
        save_visualization: Create visualization
    """
    print("="*80)
    print("🚀 FULL PIPELINE DEBUG TEST")
    print("="*80)
    
    # Performance tracking
    timings = {}
    
    # Check prerequisites
    print(f"\n✅ Prerequisites:")
    
    if not Path(settings.MODEL_PATH_YOLO).exists():
        print(f"   ❌ YOLO model not found: {settings.MODEL_PATH_YOLO}")
        return
    else:
        print(f"   ✓ YOLO model found")
    
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your-api-key-here":
        print(f"   ⚠️  Gemini API key not set (VLM extraction will be skipped)")
        use_vlm = False
    else:
        print(f"   ✓ Gemini API key configured")
        use_vlm = True
    
    # Load image
    print(f"\n📸 Loading image: {image_path}")
    try:
        image = Image.open(image_path)
        print(f"   ✓ Image loaded: {image.size[0]}x{image.size[1]} ({image.mode})")
    except Exception as e:
        print(f"   ❌ Failed to load image: {e}")
        return
    
    # Stage 1: YOLO Detection
    print(f"\n" + "="*80)
    print("STAGE 1: YOLO DETECTION")
    print("="*80)
    
    try:
        start_time = time.time()
        detector = YOLODetector()
        init_time = (time.time() - start_time) * 1000
        timings['detector_init'] = init_time
        print(f"   ✓ Detector initialized in {init_time:.2f}ms")
        
        start_time = time.time()
        detections = detector.detect(image)
        detect_time = (time.time() - start_time) * 1000
        timings['detection'] = detect_time
        
        print(f"   ✓ Detection completed in {detect_time:.2f}ms")
        print(f"   ✓ Found {len(detections)} fields")
        
        if len(detections) == 0:
            print(f"   ⚠️  No fields detected! Cannot proceed.")
            return
        
        # Show detected fields
        class_counts = {}
        for det in detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        print(f"\n   Detected fields:")
        for class_name, count in sorted(class_counts.items()):
            print(f"      - {class_name:20s}: {count}")
        
    except Exception as e:
        print(f"   ❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Stage 2: Image Enhancement
    print(f"\n" + "="*80)
    print("STAGE 2: IMAGE ENHANCEMENT")
    print("="*80)
    
    try:
        start_time = time.time()
        enhancer = PretrainedUNetEnhancer(
            model_name=settings.ENHANCEMENT_MODEL,
            encoder_name=settings.ENHANCEMENT_ENCODER,
            device=settings.DEVICE
        )
        init_time = (time.time() - start_time) * 1000
        timings['enhancer_init'] = init_time
        print(f"   ✓ Enhancer initialized in {init_time:.2f}ms")
        print(f"   Model: {settings.ENHANCEMENT_MODEL}")
        print(f"   Encoder: {settings.ENHANCEMENT_ENCODER}")
        print(f"   Method: {settings.ENHANCEMENT_METHOD}")
        
        # Crop images
        print(f"\n   Cropping {len(detections)} fields...")
        cropped_images = []
        for det in detections:
            if det.crop is not None:
                cropped_images.append(det.crop)
            else:
                # Crop from original image
                crop = crop_with_bbox(image, det.bbox)
                cropped_images.append(crop)
        
        print(f"   ✓ Cropped {len(cropped_images)} images")
        
        # Enhance batch
        print(f"\n   Enhancing crops...")
        start_time = time.time()
        enhanced_crops = enhancer.enhance_batch(
            images=cropped_images,
            detections=detections,
            method=settings.ENHANCEMENT_METHOD
        )
        enhance_time = (time.time() - start_time) * 1000
        timings['enhancement'] = enhance_time
        
        print(f"   ✓ Enhancement completed in {enhance_time:.2f}ms")
        print(f"   Average per crop: {enhance_time/len(cropped_images):.2f}ms")
        
        # Save sample enhanced crops
        output_dir = Path("output_pipeline_crops")
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n   Saving enhanced crops to: {output_dir}/")
        for idx, (ec, det) in enumerate(zip(enhanced_crops[:10], detections[:10])):
            filename = f"{idx:02d}_{det.class_name}_enhanced.jpg"
            ec.enhanced_image.save(output_dir / filename)
        print(f"   ✓ Saved {min(10, len(enhanced_crops))} sample crops")
        
    except Exception as e:
        print(f"   ❌ Enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        enhanced_crops = None
    
    # Stage 3: VLM Extraction
    if use_vlm:
        print(f"\n" + "="*80)
        print("STAGE 3: VLM EXTRACTION")
        print("="*80)
        
        try:
            start_time = time.time()
            extractor = VLMExtractor()
            init_time = (time.time() - start_time) * 1000
            timings['extractor_init'] = init_time
            print(f"   ✓ Extractor initialized in {init_time:.2f}ms")
            print(f"   Model: {settings.GEMINI_MODEL}")
            
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
            
            print(f"\n   Extracting structured data...")
            print(f"   (This may take 10-30 seconds...)")
            
            start_time = time.time()
            filename = Path(image_path).name
            extraction_result = await extractor.extract(
                original_image=image,
                detections=detections,
                enhanced_crops=extractor_crops,
                source_filename=filename
            )
            extract_time = (time.time() - start_time) * 1000
            timings['extraction'] = extract_time
            
            print(f"   ✓ Extraction completed in {extract_time:.2f}ms ({extract_time/1000:.1f}s)")
            
            # Convert to dict if it's a Pydantic model
            if hasattr(extraction_result, 'model_dump'):
                result_dict = extraction_result.model_dump()
            elif hasattr(extraction_result, 'dict'):
                result_dict = extraction_result.dict()
            else:
                result_dict = extraction_result
            
            # Show summary
            if "header" in result_dict:
                header = result_dict["header"]
                print(f"\n   📋 Header:")
                print(f"      No. KK:          {header.get('no_kk', 'N/A')}")
                print(f"      Kepala Keluarga: {header.get('kepala_keluarga', 'N/A')}")
            
            if "anggota_keluarga" in result_dict:
                members = result_dict["anggota_keluarga"]
                print(f"\n   👥 Family Members: {len(members)}")
                for i, m in enumerate(members[:3], 1):
                    print(f"      {i}. {m.get('nama_lengkap', 'N/A')} (NIK: {m.get('nik', 'N/A')})")
                if len(members) > 3:
                    print(f"      ... and {len(members)-3} more")
            
            # Save JSON
            json_path = Path(image_path).stem + "_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            print(f"\n   💾 Saved result: {json_path}")
            
        except Exception as e:
            print(f"   ❌ Extraction failed: {e}")
            import traceback
            traceback.print_exc()
            extraction_result = None
            result_dict = None
    else:
        extraction_result = None
        result_dict = None
        print(f"\n⏭️  Skipping VLM extraction (API key not set)")
    
    # Create visualization
    if save_visualization and enhanced_crops:
        print(f"\n🎨 Creating pipeline visualization...")
        try:
            save_pipeline_visualization(
                image,
                detections,
                enhanced_crops,
                result_dict
            )
        except Exception as e:
            print(f"   ⚠️  Visualization failed: {e}")
    
    # Final Summary
    print(f"\n" + "="*80)
    print("✅ PIPELINE DEBUG TEST COMPLETED")
    print("="*80)
    
    print(f"\n📈 Performance Summary:")
    total_time = sum(timings.values())
    for stage, time_ms in timings.items():
        percentage = (time_ms / total_time) * 100
        print(f"   {stage:20s}: {time_ms:8.2f}ms ({percentage:5.1f}%)")
    print(f"   {'TOTAL':20s}: {total_time:8.2f}ms ({total_time/1000:.2f}s)")
    
    # Check performance targets
    print(f"\n🎯 Performance Targets:")
    target_total = 1500  # ms
    target_yolo = 100
    target_unet = 50 * len(detections) if detections else 0
    target_vlm = 900
    
    print(f"   YOLO:       {timings.get('detection', 0):6.1f}ms / {target_yolo}ms target {'✓' if timings.get('detection', 0) < target_yolo else '⚠️'}")
    print(f"   U-Net:      {timings.get('enhancement', 0):6.1f}ms / {target_unet:.0f}ms target {'✓' if timings.get('enhancement', 0) < target_unet else '⚠️'}")
    if 'extraction' in timings:
        print(f"   VLM:        {timings.get('extraction', 0):6.1f}ms / {target_vlm}ms target {'✓' if timings.get('extraction', 0) < target_vlm else '⚠️'}")
    print(f"   Total:      {total_time:6.1f}ms / {target_total}ms target {'✓' if total_time < target_total else '⚠️'}")
    
    print(f"\n📂 Output Files:")
    print(f"   - Enhanced crops:  output_pipeline_crops/")
    if use_vlm:
        print(f"   - Extraction JSON: {Path(image_path).stem}_result.json")
    if save_visualization:
        print(f"   - Visualization:   output_pipeline_visualization.jpg")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Review visualization for pipeline flow")
    print(f"   2. Check enhanced crops quality")
    if use_vlm:
        print(f"   3. Validate extraction accuracy in JSON file")
    print(f"   4. Optimize slow stages if needed")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline_debug.py <image_path>")
        print("\nExample:")
        print("  python test_pipeline_debug.py data/raw/sample_kk.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    # Run async function
    asyncio.run(test_full_pipeline(image_path))


if __name__ == "__main__":
    main()
