"""
Validators and Utilities
File validation, image processing, JSON schema validation
"""

import io
from typing import Union, List
from pathlib import Path
import json

from PIL import Image
import numpy as np
from fastapi import UploadFile, HTTPException
import jsonschema

from src.utils.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ==================== File Validation ====================

ALLOWED_CONTENT_TYPES = [
    "image/jpeg",
    "image/jpg",
    "image/png",
    "application/pdf"
]

ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".pdf"]


def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file
    
    Args:
        file: Uploaded file
        
    Raises:
        ValueError: If file is invalid
    """
    # Check content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(
            f"Invalid file type: {file.content_type}. "
            f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}"
        )
    
    # Check file extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Invalid file extension: {ext}. "
                f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            )
    
    logger.debug(
        "File validation passed",
        extra={
            "filename": file.filename,
            "content_type": file.content_type
        }
    )


async def load_image_from_upload(file: UploadFile) -> Image.Image:
    """
    Load PIL Image from uploaded file
    
    Args:
        file: Uploaded file
        
    Returns:
        PIL Image
        
    Raises:
        ValueError: If image cannot be loaded
    """
    try:
        # Read file content
        content = await file.read()
        
        # Check file size
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File size ({size_mb:.2f} MB) exceeds maximum allowed "
                f"size ({settings.MAX_FILE_SIZE_MB} MB)"
            )
        
        # Handle PDF (extract first page)
        if file.content_type == "application/pdf":
            image = _extract_pdf_first_page(content)
        else:
            # Load image
            image = Image.open(io.BytesIO(content))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        logger.debug(
            "Image loaded successfully",
            extra={
                "size": image.size,
                "mode": image.mode,
                "format": image.format
            }
        )
        
        return image
        
    except Exception as e:
        logger.error(f"Failed to load image: {str(e)}")
        raise ValueError(f"Failed to load image: {str(e)}")


def _extract_pdf_first_page(pdf_content: bytes) -> Image.Image:
    """
    Extract first page from PDF as image
    
    Args:
        pdf_content: PDF file content
        
    Returns:
        PIL Image of first page
    """
    try:
        import fitz  # PyMuPDF
        
        # Open PDF
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Get first page
        page = doc[0]
        
        # Render page to image (300 DPI)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        doc.close()
        
        return img
        
    except ImportError:
        raise ValueError(
            "PDF support requires PyMuPDF. Install with: pip install pymupdf"
        )
    except Exception as e:
        raise ValueError(f"Failed to extract PDF: {str(e)}")


# ==================== Image Processing Utilities ====================

def resize_with_padding(
    image: Image.Image,
    target_size: int = 640,
    fill_color: tuple = (114, 114, 114)
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Resize image to target size with padding
    
    Args:
        image: Input image
        target_size: Target size (square)
        fill_color: Padding color (RGB)
        
    Returns:
        Tuple of (resized image, padding (top, right, bottom, left))
    """
    # Calculate scale
    scale = min(target_size / image.width, target_size / image.height)
    
    # New size
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    
    # Resize
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create padded image
    padded = Image.new("RGB", (target_size, target_size), fill_color)
    
    # Calculate padding
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    
    # Paste resized image
    padded.paste(resized, (pad_left, pad_top))
    
    # Calculate padding values
    pad_right = target_size - new_width - pad_left
    pad_bottom = target_size - new_height - pad_top
    
    padding = (pad_top, pad_right, pad_bottom, pad_left)
    
    return padded, padding


def crop_with_bbox(
    image: Image.Image,
    bbox: List[int]
) -> Image.Image:
    """
    Crop image using bounding box
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within image bounds
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(image.width, int(x2))
    y2 = min(image.height, int(y2))
    
    # Crop
    cropped = image.crop((x1, y1, x2, y2))
    
    return cropped


# ==================== JSON Schema Validation ====================

def load_json_schema(schema_path: str) -> dict:
    """
    Load JSON schema from file
    
    Args:
        schema_path: Path to schema file
        
    Returns:
        JSON schema dict
    """
    with open(schema_path, "r") as f:
        return json.load(f)


def validate_json_output(data: dict, schema: dict) -> bool:
    """
    Validate JSON data against schema
    
    Args:
        data: JSON data to validate
        schema: JSON schema
        
    Returns:
        True if valid
        
    Raises:
        jsonschema.ValidationError: If validation fails
    """
    jsonschema.validate(instance=data, schema=schema)
    return True


# ==================== Text Utilities ====================

def normalize_text(text: str) -> str:
    """
    Normalize extracted text
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Convert to uppercase (Indonesian KK uses uppercase)
    text = text.upper()
    
    # Handle common OCR errors (optional)
    # Example: text = text.replace("0", "O").replace("1", "I")
    
    return text


def is_empty_field(value: str) -> bool:
    """
    Check if field value is empty or placeholder
    
    Args:
        value: Field value
        
    Returns:
        True if empty
    """
    if not value:
        return True
    
    # Check common empty values
    empty_values = ["-", "_", "", "N/A", "NA", "NONE", "NULL"]
    
    return value.strip().upper() in empty_values
