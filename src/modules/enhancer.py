"""
U-Net Image Enhancer Module
Cleans and enhances cropped images for better VLM extraction
"""

from typing import List, Optional
from pathlib import Path
import time

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

from src.modules.detector import Detection
from src.utils.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ==================== U-Net Architecture ====================

class UNet(nn.Module):
    """
    U-Net architecture for image-to-image enhancement
    Input: Noisy/blurry crop (256x256)
    Output: Clean enhanced crop (256x256)
    """
    
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder_blocks.append(
                self._conv_block(in_channels, feature)
            )
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder_blocks.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                self._conv_block(feature * 2, feature)
            )
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def _conv_block(self, in_channels, out_channels):
        """Double convolution block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass"""
        skip_connections = []
        
        # Encoder
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder_blocks), 2):
            x = self.decoder_blocks[idx](x)  # Up-conv
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)  # Concatenate
            x = self.decoder_blocks[idx + 1](x)  # Conv block
        
        # Final convolution
        return torch.sigmoid(self.final_conv(x))


# ==================== Enhanced Crop ====================

class EnhancedCrop:
    """
    Enhanced crop result
    """
    def __init__(
        self,
        original: Image.Image,
        enhanced: Image.Image,
        bbox: List[int],
        class_name: str
    ):
        self.original = original
        self.enhanced = enhanced
        self.bbox = bbox
        self.class_name = class_name


# ==================== U-Net Enhancer ====================

class UNetEnhancer:
    """
    U-Net based image enhancer for KK crops
    """
    
    def __init__(self):
        """Initialize U-Net enhancer"""
        self.model_path = settings.MODEL_PATH_UNET
        self.device = settings.get_unet_device()
        self.input_size = settings.UNET_INPUT_SIZE
        self.batch_size = settings.UNET_BATCH_SIZE
        
        self.model = None
        self.transform = None
        self.inverse_transform = None
        
        self._setup_transforms()
        self._load_model()
    
    def _setup_transforms(self):
        """Setup image transformations"""
        # Transform for model input
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            # Normalize to [0, 1] - already done by ToTensor()
        ])
        
        # Inverse transform for output
        self.inverse_transform = transforms.ToPILImage()
    
    def _load_model(self):
        """Load U-Net model"""
        try:
            logger.info(
                "Loading U-Net model",
                extra={
                    "model_path": Path(self.model_path).name,
                    "device": self.device
                }
            )
            
            # Check if model file exists
            model_path = Path(self.model_path)
            if not model_path.exists():
                logger.warning(
                    f"U-Net model not found at {self.model_path}. "
                    f"Using pass-through mode (no enhancement). "
                    f"Train model using: python src/training/train_unet.py"
                )
                self.model = None
                return
            
            # Initialize model
            self.model = UNet(in_channels=3, out_channels=3)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(
                "U-Net model loaded successfully",
                extra={"device": self.device}
            )
            
            # Update metrics
            from src.utils.metrics import metrics_manager
            metrics_manager.set_model_loaded("unet", True)
            
        except Exception as e:
            logger.error(f"Failed to load U-Net model: {str(e)}", exc_info=True)
            logger.warning("Falling back to pass-through mode (no enhancement)")
            self.model = None
    
    def enhance(self, image: Image.Image) -> Image.Image:
        """
        Enhance single image
        
        Args:
            image: Input PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        # If model not loaded, return original
        if self.model is None:
            return image
        
        try:
            # Store original size
            original_size = image.size
            
            # Transform to tensor
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            # Convert back to PIL Image
            output_image = self.inverse_transform(output_tensor.squeeze(0).cpu())
            
            # Resize back to original size
            output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
            
            return output_image
            
        except Exception as e:
            logger.error(f"Enhancement failed: {str(e)}", exc_info=True)
            return image  # Return original on error
    
    def enhance_batch(
        self,
        detections: List[Detection]
    ) -> List[EnhancedCrop]:
        """
        Enhance batch of detections
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of EnhancedCrop objects
        """
        if not detections:
            return []
        
        logger.debug(f"Enhancing {len(detections)} crops")
        
        enhanced_crops = []
        
        # Process in batches
        for i in range(0, len(detections), self.batch_size):
            batch = detections[i:i + self.batch_size]
            
            for detection in batch:
                # Enhance crop
                enhanced_image = self.enhance(detection.crop)
                
                # Create enhanced crop
                enhanced_crop = EnhancedCrop(
                    original=detection.crop,
                    enhanced=enhanced_image,
                    bbox=detection.bbox,
                    class_name=detection.class_name
                )
                
                enhanced_crops.append(enhanced_crop)
        
        logger.info(f"Enhanced {len(enhanced_crops)} crops")
        
        return enhanced_crops
    
    def create_stitched_canvas(
        self,
        original_image: Image.Image,
        enhanced_crops: List[EnhancedCrop]
    ) -> Image.Image:
        """
        Create stitched canvas with enhanced crops
        This is the "enhanced version" that will be sent to VLM
        
        Args:
            original_image: Original KK image
            enhanced_crops: List of enhanced crops
            
        Returns:
            Stitched canvas with enhanced crops at original positions
        """
        # Create white canvas (same size as original)
        canvas = Image.new("RGB", original_image.size, (255, 255, 255))
        
        # Paste each enhanced crop at its original position
        for crop in enhanced_crops:
            # Get bbox
            x1, y1, x2, y2 = crop.bbox
            
            # Resize enhanced crop to match bbox size
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            if bbox_width > 0 and bbox_height > 0:
                resized_crop = crop.enhanced.resize(
                    (bbox_width, bbox_height),
                    Image.Resampling.LANCZOS
                )
                
                # Paste on canvas
                canvas.paste(resized_crop, (x1, y1))
        
        return canvas
    
    def apply_traditional_enhancement(self, image: Image.Image) -> Image.Image:
        """
        Apply traditional image enhancement techniques
        Fallback method when U-Net model is not available
        
        Args:
            image: Input PIL Image
            
        Returns:
            Enhanced PIL Image using traditional methods
        """
        try:
            import cv2
            from skimage import exposure
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive thresholding (binarization)
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
            
            # Morphological operations (remove lines)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(morphed, cv2.COLOR_GRAY2RGB)
            
            # Convert to PIL Image
            return Image.fromarray(enhanced)
            
        except ImportError:
            logger.warning("OpenCV or scikit-image not available for traditional enhancement")
            return image
        except Exception as e:
            logger.error(f"Traditional enhancement failed: {str(e)}")
            return image
