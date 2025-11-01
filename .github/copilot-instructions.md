# KK-OCR v2 Pipeline Project Instructions

This project implements a three-stage pipeline for extracting structured data from Indonesian Family Card (Kartu Keluarga) documents using YOLO detection, U-Net image enhancement, and Gemini VLM extraction.

## Project Overview
- **Language**: Python 3.10+
- **Framework**: FastAPI
- **ML Models**: YOLOv8 (detection), U-Net (enhancement), Gemini 1.5 Pro (extraction)
- **Deployment**: Docker containerization

## Architecture
The system processes documents through a 3-stage pipeline:
1. **YOLO Detection**: Detects 22 field classes with mAP@0.5-0.95 = 0.886
2. **U-Net Enhancement**: Cleans and enhances cropped images (denoising, binarization, line removal)
3. **VLM Extraction**: Extracts structured JSON data using Gemini API

## Development Guidelines

### Code Structure
- All source code in `src/` directory
- Modular design: separate files for each component (detector, enhancer, extractor)
- Models stored in `models/` directory
- Configuration in `configs/` directory
- Tests in `tests/` directory

### Security Requirements
- **NO PII in logs**: Never log NIK, names, or personal data
- API authentication required (Bearer token)
- Use environment variables for sensitive data

### Performance Requirements
- End-to-end latency: < 1500ms (P95)
- YOLO inference: < 100ms
- U-Net enhancement: < 50ms per crop
- VLM API call: < 900ms

### Output Format
- Strict JSON schema compliance (see PRD)
- Three sections: metadata, header, anggota_keluarga, footer
- Field-level exact match accuracy > 95%
- Character Error Rate (CER) < 2% for NIK and names

### Model Management
- Version all model artifacts (.pt files, prompts)
- Use model registry or DVC for versioning
- Include model versions in output metadata

### API Design
- FastAPI with async support
- Endpoint: `POST /v2/extract/kk`
- Input: multipart/form-data (JPEG, PNG, PDF)
- Output: JSON (200) or error JSON (4XX, 5XX)
- Prometheus metrics at `/metrics`

### Docker Requirements
- Containerize entire application
- Stateless design for horizontal scaling
- Support GPU (CUDA) for inference
- Multi-stage build for optimization

## Key Technologies
- `fastapi` - API framework
- `ultralytics` - YOLO detection
- `torch` - U-Net model
- `google-generativeai` - Gemini API
- `pillow` - Image processing
- `pydantic` - Data validation
- `prometheus-client` - Metrics
- `uvicorn` - ASGI server

## Testing Requirements
- Unit tests for each module
- Integration tests for full pipeline
- Test fixtures with sample KK images
- Mock VLM responses for testing

## Documentation Requirements
- README with setup instructions
- API documentation (OpenAPI/Swagger)
- Model training documentation
- Deployment guide
