"""
Pretrained U-Net Enhancer Debug Script
Test and visualize enhancement results
"""

import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from src.modules.enhancer_pretrained import PretrainedUNetEnhancer
from src.modules.detector import Detection
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def create_comparison_grid(original, enhanced, title="Enhancement Comparison", save_path="output_enhancement_comparison.jpg"):
    """
    Create side-by-side comparison of original and enhanced images
    
    Args:
        original: Original PIL Image
        enhanced: Enhanced PIL Image
        title: Plot title
        save_path: Path to save comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original, cmap='gray' if original.mode == 'L' else None)
    axes[0].set_title("Original", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Enhanced image
    axes[1].imshow(enhanced, cmap='gray' if enhanced.mode == 'L' else None)
    axes[1].set_title("Enhanced", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved comparison: {save_path}")


def analyze_image_quality(image: Image.Image, label: str = "Image"):
    """
    Analyze image quality metrics
    
    Args:
        image: PIL Image
        label: Label for the image
    """
    # Convert to numpy
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        # Convert to grayscale for analysis
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # Calculate metrics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    min_intensity = np.min(gray)
    max_intensity = np.max(gray)
    contrast = max_intensity - min_intensity
    
    # Calculate sharpness (Laplacian variance)
    from scipy import ndimage
    laplacian = ndimage.laplace(gray)
    sharpness = np.var(laplacian)
    
    print(f"\n   📊 {label} Quality Metrics:")
    print(f"      Size:       {image.size[0]}x{image.size[1]}")
    print(f"      Mode:       {image.mode}")
    print(f"      Mean:       {mean_intensity:.2f}")
    print(f"      Std:        {std_intensity:.2f}")
    print(f"      Range:      [{min_intensity:.0f}, {max_intensity:.0f}]")
    print(f"      Contrast:   {contrast:.2f}")
    print(f"      Sharpness:  {sharpness:.2f}")
    
    return {
        'mean': mean_intensity,
        'std': std_intensity,
        'contrast': contrast,
        'sharpness': sharpness
    }


def test_enhancement_methods(image_path: str):
    """
    Test all enhancement methods and compare results
    
    Args:
        image_path: Path to test image
    """
    print("="*80)
    print("🎨 PRETRAINED U-NET ENHANCER DEBUG TEST")
    print("="*80)
    
    print(f"\n📋 Configuration:")
    print(f"   Model:      {settings.ENHANCEMENT_MODEL}")
    print(f"   Encoder:    {settings.ENHANCEMENT_ENCODER}")
    print(f"   Method:     {settings.ENHANCEMENT_METHOD}")
    print(f"   Device:     {settings.DEVICE}")
    
    # Load image
    print(f"\n📸 Loading image: {image_path}")
    try:
        image = Image.open(image_path)
        print(f"   ✓ Image loaded: {image.size[0]}x{image.size[1]} ({image.mode})")
    except Exception as e:
        print(f"   ❌ Failed to load image: {e}")
        sys.exit(1)
    
    # Analyze original image
    analyze_image_quality(image, "Original")
    
    # Create dummy detection for full image
    dummy_detection = Detection(
        bbox=[0, 0, image.size[0], image.size[1]],
        class_name="test_field",
        confidence=1.0,
        crop=image
    )
    
    # Initialize enhancer
    print(f"\n🚀 Initializing Enhancer...")
    try:
        start_time = time.time()
        enhancer = PretrainedUNetEnhancer(
            model_name=settings.ENHANCEMENT_MODEL,
            encoder_name=settings.ENHANCEMENT_ENCODER,
            device=settings.DEVICE
        )
        init_time = (time.time() - start_time) * 1000
        print(f"   ✓ Enhancer initialized in {init_time:.2f}ms")
    except Exception as e:
        print(f"   ❌ Failed to initialize enhancer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test each enhancement method
    methods = ["classical", "deep", "hybrid"]
    results = {}
    timings = {}
    
    print(f"\n🔬 Testing Enhancement Methods:")
    print("="*80)
    
    for method in methods:
        print(f"\n[{method.upper()}]")
        try:
            start_time = time.time()
            result = enhancer.enhance(image, dummy_detection, method=method)
            elapsed = (time.time() - start_time) * 1000
            
            print(f"   ✓ Enhancement completed in {elapsed:.2f}ms")
            results[method] = result.enhanced_image
            timings[method] = elapsed
            
            # Analyze enhanced image
            metrics = analyze_image_quality(result.enhanced_image, f"{method.capitalize()} Enhanced")
            
            # Save individual result
            save_path = f"output_enhanced_{method}.jpg"
            result.enhanced_image.save(save_path)
            print(f"   💾 Saved: {save_path}")
            
            # Create comparison
            comparison_path = f"output_comparison_{method}.jpg"
            create_comparison_grid(image, result.enhanced_image, 
                                   f"{method.capitalize()} Enhancement", 
                                   comparison_path)
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Create combined comparison
    if len(results) >= 2:
        print(f"\n🎨 Creating combined comparison...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        axes = axes.flatten()
        
        # Original
        axes[0].imshow(image, cmap='gray' if image.mode == 'L' else None)
        axes[0].set_title("Original", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Enhanced versions
        for idx, method in enumerate(methods):
            if method in results:
                axes[idx+1].imshow(results[method], cmap='gray' if results[method].mode == 'L' else None)
                axes[idx+1].set_title(f"{method.capitalize()} ({timings[method]:.1f}ms)", 
                                     fontsize=14, fontweight='bold')
                axes[idx+1].axis('off')
        
        plt.suptitle("Enhancement Method Comparison", fontsize=16, fontweight='bold')
        plt.tight_layout()
        combined_path = "output_all_methods_comparison.jpg"
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved combined comparison: {combined_path}")
    
    # Performance summary
    print("\n" + "="*80)
    print("✅ ENHANCEMENT DEBUG TEST COMPLETED")
    print("="*80)
    
    print(f"\n📈 Performance Summary:")
    print(f"   Initialization: {init_time:.2f}ms")
    for method in methods:
        if method in timings:
            print(f"   {method.capitalize():10s}:  {timings[method]:6.2f}ms")
    
    # Speed ranking
    sorted_methods = sorted(timings.items(), key=lambda x: x[1])
    print(f"\n🏆 Speed Ranking:")
    for rank, (method, time_ms) in enumerate(sorted_methods, 1):
        print(f"   {rank}. {method.capitalize():10s}: {time_ms:.2f}ms")
    
    print(f"\n📂 Output Files:")
    print(f"   - Individual results:     output_enhanced_<method>.jpg")
    print(f"   - Comparisons:            output_comparison_<method>.jpg")
    print(f"   - Combined comparison:    output_all_methods_comparison.jpg")
    
    print(f"\n💡 Recommendations:")
    fastest_method = sorted_methods[0][0]
    print(f"   - Fastest method:  {fastest_method} ({sorted_methods[0][1]:.2f}ms)")
    print(f"   - Best quality:    hybrid (combines classical + deep learning)")
    print(f"   - CPU-only:        classical (no GPU needed)")


def test_batch_enhancement(image_paths: list):
    """
    Test batch enhancement with multiple images
    
    Args:
        image_paths: List of image paths
    """
    print("="*80)
    print("📦 BATCH ENHANCEMENT DEBUG TEST")
    print("="*80)
    
    print(f"\n📸 Loading {len(image_paths)} images...")
    images = []
    detections = []
    
    for idx, path in enumerate(image_paths):
        try:
            img = Image.open(path)
            images.append(img)
            det = Detection(
                bbox=[0, 0, img.size[0], img.size[1]],
                class_name=f"field_{idx}",
                confidence=1.0,
                crop=img
            )
            detections.append(det)
            print(f"   ✓ [{idx+1}] {Path(path).name} ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            print(f"   ❌ [{idx+1}] Failed to load {path}: {e}")
    
    if len(images) == 0:
        print("❌ No images loaded!")
        return
    
    # Initialize enhancer
    print(f"\n🚀 Initializing Enhancer...")
    enhancer = PretrainedUNetEnhancer(device=settings.DEVICE)
    print(f"   ✓ Ready")
    
    # Batch process
    print(f"\n⚡ Processing batch with hybrid method...")
    start_time = time.time()
    results = enhancer.enhance_batch(images, detections, method="hybrid")
    elapsed = (time.time() - start_time) * 1000
    
    print(f"   ✓ Batch completed in {elapsed:.2f}ms")
    print(f"   Average per image: {elapsed/len(images):.2f}ms")
    
    # Save results
    output_dir = Path("output_batch_enhanced")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving enhanced images to: {output_dir}/")
    for idx, result in enumerate(results):
        save_path = output_dir / f"enhanced_{idx:02d}.jpg"
        result.enhanced_image.save(save_path)
        print(f"   ✓ {save_path.name}")
    
    print(f"\n✅ Batch enhancement completed!")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image:  python test_enhancer_debug.py <image_path>")
        print("  Batch mode:    python test_enhancer_debug.py <image1> <image2> ...")
        print("\nExample:")
        print("  python test_enhancer_debug.py data/raw/sample_kk.jpg")
        print("  python test_enhancer_debug.py output_crops_yolo/*.jpg")
        sys.exit(1)
    
    image_paths = sys.argv[1:]
    
    # Check if images exist
    valid_paths = []
    for path in image_paths:
        if Path(path).exists():
            valid_paths.append(path)
        else:
            print(f"⚠️  Warning: Image not found: {path}")
    
    if len(valid_paths) == 0:
        print("❌ No valid images found!")
        sys.exit(1)
    
    if len(valid_paths) == 1:
        test_enhancement_methods(valid_paths[0])
    else:
        test_batch_enhancement(valid_paths)


if __name__ == "__main__":
    main()
