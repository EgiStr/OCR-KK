"""
VLM Extractor Module
Uses Google Gemini to extract structured data from KK documents
"""

import json
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import io

from PIL import Image
import google.generativeai as genai

from src.modules.detector import Detection
from src.modules.enhancer import EnhancedCrop
from src.api.models import (
    KKExtractionResponse, Metadata, HeaderData,
    FamilyMember, FooterData, SignatureInfo
)
from src.utils.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ==================== Prompt Templates ====================

SYSTEM_PROMPT = """You are an expert data extraction specialist for Indonesian Family Card (Kartu Keluarga/KK) documents.

You will receive TWO images:
1. ORIGINAL IMAGE - Use this to understand STRUCTURE, LAYOUT, and ROW ASSOCIATIONS
2. ENHANCED IMAGE - Use this to READ TEXT clearly (it has been cleaned and enhanced)

Your task:
1. Analyze the document structure from IMAGE 1 (original)
2. Read text values from IMAGE 2 (enhanced/cleaned)
3. Associate family member data by ROWS - each row represents one person
4. Extract all data into the specified JSON format
5. Handle empty fields with null or "-"
6. Ensure NIK (16 digits) and dates (DD-MM-YYYY format) are correctly extracted

CRITICAL RULES:
- Each family member is ONE ROW in the table
- nama_ayah and nama_ibu must match the row position (not the header kepala_keluarga)
- Use IMAGE 2 for reading text (it's cleaner)
- Use IMAGE 1 for understanding which fields belong to which family member
- Output ONLY valid JSON, no explanations
"""

EXTRACTION_PROMPT = """Extract all data from this Kartu Keluarga document.

Required output format:
{
  "metadata": {
    "processing_timestamp": "<ISO 8601 timestamp>",
    "model_version_yolo": "v1.0.0",
    "model_version_unet": "v1.0.0",
    "model_version_vlm": "gemini-1.5-pro",
    "source_file": "<filename>"
  },
  "header": {
    "no_kk": "<16 digit number>",
    "kepala_keluarga": "<name>",
    "alamat": "<address>",
    "rt": "<RT>",
    "rw": "<RW>",
    "desa": "<desa/kelurahan>",
    "kecamatan": "<kecamatan>",
    "kabupaten_kota": "<kabupaten/kota>",
    "provinsi": "<provinsi>",
    "kode_pos": "<postal code>",
    "tanggal_pembuatan": "<DD-MM-YYYY>"
  },
  "anggota_keluarga": [
    {
      "nama_lengkap": "<full name>",
      "nik": "<16 digit NIK>",
      "jenis_kelamin": "LAKI-LAKI or PEREMPUAN",
      "tempat_lahir": "<place>",
      "tanggal_lahir": "<DD-MM-YYYY>",
      "agama": "<religion>",
      "pendidikan": "<education level>",
      "jenis_pekerjaan": "<occupation>",
      "status_perkawinan": "<marital status>",
      "status_keluarga": "<family status>",
      "kewarganegaraan": "WNI or WNA",
      "no_paspor": "<passport number or '-'>",
      "no_KITAP": "<KITAP number or '-'>",
      "nama_ayah": "<father name from this row>",
      "nama_ibu": "<mother name from this row>",
      "golongan_darah": "<blood type or null>",
      "tanggal_perkawinan": "<DD-MM-YYYY or null>"
    }
  ],
  "footer": {
    "tanda_tangan_kepala_keluarga": {
      "terdeteksi": true/false,
      "text": "<signature text or null>"
    },
    "tanda_tangan_pejabat": {
      "terdeteksi": true/false,
      "text": "<official signature text or null>"
    }
  }
}

Extract ALL family members (anggota_keluarga) - there may be multiple rows.
Remember: Use IMAGE 1 for layout, IMAGE 2 for reading text.
Output ONLY the JSON, nothing else.
"""


# ==================== VLM Extractor ====================

class VLMExtractor:
    """
    Gemini VLM-based data extractor
    """
    
    def __init__(self):
        """Initialize VLM extractor"""
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.timeout = settings.GEMINI_TIMEOUT
        self.max_retries = settings.GEMINI_MAX_RETRIES
        
        self.model = None
        self._configure_gemini()
    
    def _configure_gemini(self):
        """Configure Gemini API"""
        try:
            logger.info("Configuring Gemini API")
            
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY not set. Please set it in .env file."
                )
            
            # Configure API
            genai.configure(api_key=self.api_key)
            
            # Initialize model
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.1,  # Low temperature for factual extraction
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            
            logger.info(
                "Gemini API configured successfully",
                extra={"model": self.model_name}
            )
            
            # Update metrics
            from src.utils.metrics import metrics_manager
            metrics_manager.set_model_loaded("vlm", True)
            
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to configure Gemini API: {str(e)}")
    
    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes"""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    
    async def extract(
        self,
        original_image: Image.Image,
        detections: List[Detection],
        enhanced_crops: List[EnhancedCrop],
        source_filename: str
    ) -> KKExtractionResponse:
        """
        Extract structured data from KK document
        
        Args:
            original_image: Original KK image
            detections: YOLO detections
            enhanced_crops: U-Net enhanced crops
            source_filename: Original filename
            
        Returns:
            KKExtractionResponse with structured data
        """
        if self.model is None:
            raise RuntimeError("Gemini model not configured")
        
        try:
            logger.info("Starting VLM extraction")
            
            # Create stitched canvas with enhanced crops
            from src.modules.enhancer import UNetEnhancer
            enhancer = UNetEnhancer()
            enhanced_canvas = enhancer.create_stitched_canvas(
                original_image, enhanced_crops
            )
            
            # Prepare prompt
            prompt = f"{SYSTEM_PROMPT}\n\n{EXTRACTION_PROMPT}"
            
            # Prepare images for Gemini
            # Send both original and enhanced
            images = [
                {
                    "mime_type": "image/png",
                    "data": self._image_to_bytes(original_image)
                },
                {
                    "mime_type": "image/png",
                    "data": self._image_to_bytes(enhanced_canvas)
                }
            ]
            
            # Call Gemini API with retry logic
            response_text = await self._call_gemini_with_retry(
                prompt=prompt,
                images=images
            )
            
            # Parse JSON response
            extracted_data = self._parse_response(response_text, source_filename)
            
            logger.info("VLM extraction completed successfully")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"VLM extraction failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"VLM extraction failed: {str(e)}")
    
    async def extract_direct(
        self,
        image: Image.Image,
        source_filename: str
    ) -> KKExtractionResponse:
        """
        Extract structured data directly from image without detection.
        Used by VLM-only pipeline.
        
        Args:
            image: Original KK image
            source_filename: Original filename
            
        Returns:
            KKExtractionResponse with structured data
        """
        if self.model is None:
            raise RuntimeError("Gemini model not configured")
        
        try:
            logger.info("Starting direct VLM extraction (no detection)")
            
            # Simplified prompt for direct extraction
            direct_prompt = """You are an expert data extraction specialist for Indonesian Family Card (Kartu Keluarga/KK) documents.

Analyze this document image and extract all data into structured JSON format.

Your task:
1. Identify the document type and structure
2. Extract header information (No. KK, kepala keluarga, alamat, etc.)
3. Extract all family members from the table rows
4. Extract footer information (signatures)

""" + EXTRACTION_PROMPT
            
            # Prepare single image for Gemini
            images = [
                {
                    "mime_type": "image/png",
                    "data": self._image_to_bytes(image)
                }
            ]
            
            # Call Gemini API
            response_text = await self._call_gemini_with_retry(
                prompt=direct_prompt,
                images=images
            )
            
            # Parse JSON response
            extracted_data = self._parse_response(response_text, source_filename)
            
            logger.info("Direct VLM extraction completed successfully")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Direct VLM extraction failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Direct VLM extraction failed: {str(e)}")
    
    async def extract_with_detections(
        self,
        image: Image.Image,
        detections: List[Detection],
        source_filename: str
    ) -> KKExtractionResponse:
        """
        Extract structured data using YOLO detections for context.
        Used by YOLO+VLM pipeline (recommended).
        
        Args:
            image: Original KK image
            detections: YOLO detection results with bounding boxes
            source_filename: Original filename
            
        Returns:
            KKExtractionResponse with structured data
        """
        if self.model is None:
            raise RuntimeError("Gemini model not configured")
        
        try:
            logger.info(
                "Starting VLM extraction with detection context",
                extra={"num_detections": len(detections)}
            )
            
            # Create annotated image with bounding boxes
            annotated_image = self._annotate_image_with_detections(image, detections)
            
            # Create detection context for prompt
            detection_context = self._create_detection_context(detections)
            
            # Enhanced prompt with detection info
            detection_prompt = f"""{SYSTEM_PROMPT}

DETECTION RESULTS:
The following fields have been detected in the document:
{detection_context}

Use these detections to help associate data with correct rows.

{EXTRACTION_PROMPT}"""
            
            # Prepare images: original + annotated
            images = [
                {
                    "mime_type": "image/png",
                    "data": self._image_to_bytes(image)
                },
                {
                    "mime_type": "image/png",
                    "data": self._image_to_bytes(annotated_image)
                }
            ]
            
            # Call Gemini API
            response_text = await self._call_gemini_with_retry(
                prompt=detection_prompt,
                images=images
            )
            
            # Parse JSON response
            extracted_data = self._parse_response(response_text, source_filename)
            
            logger.info("VLM extraction with detections completed successfully")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"VLM extraction with detections failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"VLM extraction with detections failed: {str(e)}")
    
    def _annotate_image_with_detections(
        self,
        image: Image.Image,
        detections: List[Detection]
    ) -> Image.Image:
        """
        Draw bounding boxes and labels on image for VLM context.
        
        Args:
            image: Original image
            detections: List of detections
            
        Returns:
            Annotated PIL Image
        """
        from PIL import ImageDraw, ImageFont
        
        # Create a copy to draw on
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Try to use a font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Color palette for different classes
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
            "#FFA500", "#800080", "#008080", "#FFD700", "#DC143C", "#00CED1"
        ]
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            color = colors[i % len(colors)]
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            label = f"{det.class_name}"
            draw.text((x1, y1 - 15), label, fill=color, font=font)
        
        return annotated
    
    def _create_detection_context(self, detections: List[Detection]) -> str:
        """
        Create textual context from detections for the prompt.
        
        Args:
            detections: List of detections
            
        Returns:
            Formatted string with detection info
        """
        if not detections:
            return "No specific fields detected."
        
        # Group by class
        class_groups = {}
        for det in detections:
            if det.class_name not in class_groups:
                class_groups[det.class_name] = []
            class_groups[det.class_name].append(det)
        
        lines = []
        for class_name, dets in sorted(class_groups.items()):
            positions = [f"({int(d.bbox[0])},{int(d.bbox[1])})" for d in dets]
            lines.append(f"- {class_name}: {len(dets)} detected at positions {', '.join(positions)}")
        
        return "\n".join(lines)

    
    async def _call_gemini_with_retry(
        self,
        prompt: str,
        images: List[Dict[str, Any]]
    ) -> str:
        """
        Call Gemini API with retry logic
        
        Args:
            prompt: Text prompt
            images: List of image dicts
            
        Returns:
            Response text
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Calling Gemini API (attempt {attempt + 1}/{self.max_retries})"
                )
                
                # Prepare content
                content = [prompt]
                for img in images:
                    content.append(img)
                
                # Generate response
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.model.generate_content,
                        content
                    ),
                    timeout=self.timeout
                )
                
                # Extract text
                response_text = response.text
                
                logger.debug("Gemini API call successful")
                
                return response_text
                
            except asyncio.TimeoutError:
                last_error = "Gemini API timeout"
                logger.warning(f"Attempt {attempt + 1} timed out")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}"
                )
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed
        raise RuntimeError(
            f"Gemini API call failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
    def _parse_response(
        self,
        response_text: str,
        source_filename: str
    ) -> KKExtractionResponse:
        """
        Parse Gemini response into structured format
        
        Args:
            response_text: Raw response from Gemini
            source_filename: Original filename
            
        Returns:
            KKExtractionResponse
        """
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned_text = response_text.strip()
            
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            
            cleaned_text = cleaned_text.strip()
            
            # Parse JSON
            data = json.loads(cleaned_text)
            
            # Update metadata
            if "metadata" not in data:
                data["metadata"] = {}
            
            data["metadata"]["processing_timestamp"] = datetime.utcnow().isoformat() + "Z"
            data["metadata"]["source_file"] = source_filename
            data["metadata"]["model_version_yolo"] = "v1.0.0"
            data["metadata"]["model_version_unet"] = "v1.0.0"
            data["metadata"]["model_version_vlm"] = self.model_name
            
            # Validate and create response model
            response = KKExtractionResponse(**data)
            
            return response
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Response text: {response_text[:500]}")
            raise ValueError(f"Invalid JSON response from VLM: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to parse response: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to parse VLM response: {str(e)}")
    
    def validate_extraction(self, response: KKExtractionResponse) -> bool:
        """
        Validate extracted data
        
        Args:
            response: Extraction response
            
        Returns:
            True if valid
        """
        try:
            # Check required fields
            if not response.header.no_kk:
                logger.warning("Missing no_kk in header")
                return False
            
            if not response.anggota_keluarga:
                logger.warning("No family members extracted")
                return False
            
            # Validate NIKs (should be 16 digits)
            for member in response.anggota_keluarga:
                if member.nik and len(member.nik) != 16:
                    logger.warning(
                        f"Invalid NIK length: {member.nik} "
                        f"(expected 16, got {len(member.nik)})"
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False
