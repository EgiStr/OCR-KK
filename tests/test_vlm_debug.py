"""
VLM (Gemini) Extractor Debug Script
Test and debug VLM extraction
"""

import sys
import json
import time
import asyncio
from pathlib import Path
from PIL import Image

from src.modules.extractor import VLMExtractor
from src.modules.detector import Detection
from src.modules.enhancer_pretrained import PretrainedUNetEnhancer, EnhancedCrop
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def print_extraction_result(result: dict, verbose: bool = True):
    """
    Pretty print extraction result
    
    Args:
        result: Extraction result dictionary
        verbose: Print full details
    """
    print("\n" + "="*80)
    print("📄 EXTRACTION RESULT")
    print("="*80)
    
    # Metadata
    if "metadata" in result:
        meta = result["metadata"]
        print(f"\n📋 Metadata:")
        print(f"   Document Type: {meta.get('document_type', 'N/A')}")
        print(f"   Confidence:    {meta.get('confidence_score', 'N/A')}")
        print(f"   Processing:    {meta.get('processing_time_ms', 'N/A')}ms")
    
    # Header
    if "header" in result:
        header = result["header"]
        print(f"\n📌 Header:")
        for key, value in header.items():
            if value:
                print(f"   {key:20s}: {value}")
    
    # Family Members
    if "anggota_keluarga" in result:
        members = result["anggota_keluarga"]
        print(f"\n👥 Family Members ({len(members)}):")
        for idx, member in enumerate(members, 1):
            print(f"\n   [{idx}] {member.get('nama_lengkap', 'N/A')}")
            if verbose:
                for key, value in member.items():
                    if key != 'nama_lengkap' and value:
                        print(f"       {key:20s}: {value}")
            else:
                print(f"       NIK:     {member.get('nik', 'N/A')}")
                print(f"       Status:  {member.get('status_keluarga', 'N/A')}")
    
    # Footer
    if "footer" in result:
        footer = result["footer"]
        print(f"\n📝 Footer:")
        for key, value in footer.items():
            if value:
                print(f"   {key:20s}: {value}")


async def test_vlm_with_image(image_path: str, save_json: bool = True):
    """
    Test VLM extraction with a single image
    
    Args:
        image_path: Path to KK image
        save_json: Save result as JSON
    """
    print("="*80)
    print("🤖 VLM (GEMINI) EXTRACTOR DEBUG TEST")
    print("="*80)
    
    # Check API key
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your-api-key-here":
        print("❌ ERROR: GEMINI_API_KEY not set!")
        print("   Please set GEMINI_API_KEY in .env file")
        sys.exit(1)
    
    print(f"\n📋 Configuration:")
    print(f"   Model:      {settings.GEMINI_MODEL}")
    print(f"   Timeout:    {settings.GEMINI_TIMEOUT}s")
    print(f"   Max Retry:  {settings.GEMINI_MAX_RETRIES}")
    print(f"   API Key:    {'*' * 20}{settings.GEMINI_API_KEY[-8:]}")
    
    # Load image
    print(f"\n📸 Loading image: {image_path}")
    try:
        image = Image.open(image_path)
        print(f"   ✓ Image loaded: {image.size[0]}x{image.size[1]} ({image.mode})")
    except Exception as e:
        print(f"   ❌ Failed to load image: {e}")
        sys.exit(1)
    
    # Create dummy detection (full image)
    detection = Detection(
        bbox=[0, 0, image.size[0], image.size[1]],
        class_name="full_document",
        confidence=1.0,
        crop=image
    )
    
    # Create enhanced crop (for VLM extraction)
    print(f"\n🎨 Enhancing image for better OCR...")
    try:
        enhancer = PretrainedUNetEnhancer(device=settings.DEVICE)
        enhanced_result = enhancer.enhance(image, detection, method="hybrid")
        
        # Convert to extractor-compatible EnhancedCrop format
        from src.modules.enhancer import EnhancedCrop as ExtractorEnhancedCrop
        enhanced_crop = ExtractorEnhancedCrop(
            original=image,
            enhanced=enhanced_result.enhanced_image,
            bbox=[0, 0, image.size[0], image.size[1]],
            class_name="full_document"
        )
        print(f"   ✓ Image enhanced")
    except Exception as e:
        print(f"   ⚠️  Enhancement failed, using original: {e}")
        from src.modules.enhancer import EnhancedCrop as ExtractorEnhancedCrop
        enhanced_crop = ExtractorEnhancedCrop(
            original=image,
            enhanced=image,
            bbox=[0, 0, image.size[0], image.size[1]],
            class_name="full_document"
        )
    
    # Initialize extractor
    print(f"\n🚀 Initializing VLM Extractor...")
    try:
        start_time = time.time()
        extractor = VLMExtractor()
        init_time = (time.time() - start_time) * 1000
        print(f"   ✓ Extractor initialized in {init_time:.2f}ms")
    except Exception as e:
        print(f"   ❌ Failed to initialize extractor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Extract data
    print(f"\n🔍 Extracting structured data from image...")
    print(f"   (This may take 5-30 seconds depending on image complexity)")
    
    try:
        start_time = time.time()
        filename = Path(image_path).name
        result = await extractor.extract(
            original_image=image,
            detections=[detection],
            enhanced_crops=[enhanced_crop],
            source_filename=filename
        )
        extract_time = (time.time() - start_time) * 1000
        
        print(f"   ✓ Extraction completed in {extract_time:.2f}ms ({extract_time/1000:.1f}s)")
        
        # Convert to dict if it's a Pydantic model
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
        else:
            result_dict = result
        
        # Print result
        print_extraction_result(result_dict, verbose=True)
        
        # Validate result
        print(f"\n✅ Validation:")
        
        # Check required fields
        required_sections = ["metadata", "header", "anggota_keluarga", "footer"]
        for section in required_sections:
            if section in result_dict:
                print(f"   ✓ {section:20s}: Present")
            else:
                print(f"   ⚠️  {section:20s}: Missing")
        
        # Check data completeness
        if "header" in result_dict:
            header_fields = ["no_kk", "kepala_keluarga", "alamat", "provinsi"]
            filled = sum(1 for f in header_fields if result_dict["header"].get(f))
            print(f"   📊 Header completeness: {filled}/{len(header_fields)} fields")
        
        if "anggota_keluarga" in result_dict:
            members = result_dict["anggota_keluarga"]
            if len(members) > 0:
                member_fields = ["nik", "nama_lengkap", "jenis_kelamin", "tanggal_lahir"]
                avg_filled = sum(
                    sum(1 for f in member_fields if m.get(f)) 
                    for m in members
                ) / (len(members) * len(member_fields))
                print(f"   📊 Member completeness: {avg_filled*100:.1f}%")
        
        # Save as JSON
        if save_json:
            json_path = Path(image_path).stem + "_extracted.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved extraction result: {json_path}")
        
        # Summary
        print("\n" + "="*80)
        print("✅ VLM DEBUG TEST COMPLETED")
        print("="*80)
        
        print(f"\n📈 Performance Summary:")
        print(f"   Initialization: {init_time:.2f}ms")
        print(f"   Extraction:     {extract_time:.2f}ms ({extract_time/1000:.1f}s)")
        print(f"   Total:          {init_time + extract_time:.2f}ms")
        
        print(f"\n💡 Tips:")
        print(f"   - Check JSON file for full structured data")
        print(f"   - Compare with original image for accuracy")
        print(f"   - Note any fields that were missed or incorrect")
        
        return result_dict
        
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_vlm_with_crops(crops_dir: str):
    """
    Test VLM extraction with individual cropped fields
    
    Args:
        crops_dir: Directory containing cropped field images
    """
    print("="*80)
    print("🔬 VLM FIELD-BY-FIELD EXTRACTION TEST")
    print("="*80)
    
    crops_path = Path(crops_dir)
    if not crops_path.exists():
        print(f"❌ Directory not found: {crops_dir}")
        sys.exit(1)
    
    # Get all image files
    image_files = sorted(crops_path.glob("*.jpg")) + sorted(crops_path.glob("*.png"))
    
    if len(image_files) == 0:
        print(f"❌ No images found in: {crops_dir}")
        sys.exit(1)
    
    print(f"\n📁 Found {len(image_files)} crop images")
    
    # Initialize extractor
    print(f"\n🚀 Initializing VLM Extractor...")
    extractor = VLMExtractor()
    
    # Process each crop
    results = {}
    total_time = 0
    
    print(f"\n🔍 Extracting text from each crop:")
    print("="*80)
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {img_path.name}")
        
        try:
            # Load image
            image = Image.open(img_path)
            print(f"   Size: {image.size[0]}x{image.size[1]}")
            
            # Extract field name from filename
            field_name = img_path.stem.split('_', 1)[1] if '_' in img_path.stem else "unknown"
            
            # Create detection
            detection = Detection(
                bbox=[0, 0, image.size[0], image.size[1]],
                class_name=field_name,
                confidence=1.0,
                crop=image
            )
            
            # Extract
            start_time = time.time()
            result = extractor.extract(image, [detection])
            elapsed = (time.time() - start_time) * 1000
            total_time += elapsed
            
            print(f"   Time: {elapsed:.2f}ms")
            print(f"   Result: {json.dumps(result, ensure_ascii=False)[:100]}...")
            
            results[img_path.name] = result
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Save combined results
    json_path = "extracted_fields.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "="*80)
    print(f"✅ Processed {len(results)}/{len(image_files)} crops")
    print(f"💾 Saved results: {json_path}")
    print(f"⏱️  Total time: {total_time:.2f}ms")
    print(f"📊 Average per crop: {total_time/len(results):.2f}ms")
    print("="*80)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Full image:     python test_vlm_debug.py <image_path>")
        print("  Cropped fields: python test_vlm_debug.py --crops <crops_directory>")
        print("\nExample:")
        print("  python test_vlm_debug.py data/raw/sample_kk.jpg")
        print("  python test_vlm_debug.py --crops output_crops_yolo/")
        sys.exit(1)
    
    if sys.argv[1] == "--crops":
        if len(sys.argv) < 3:
            print("❌ Please specify crops directory")
            sys.exit(1)
        test_vlm_with_crops(sys.argv[2])
    else:
        image_path = sys.argv[1]
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            sys.exit(1)
        # Run async function
        asyncio.run(test_vlm_with_image(image_path))


if __name__ == "__main__":
    main()
