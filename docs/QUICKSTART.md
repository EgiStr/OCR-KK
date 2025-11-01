# Quick Start Guide

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Docker & Docker Compose (for containerized deployment)
- Gemini API key from Google AI Studio

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd OCR-KK
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and set your values
nano .env
```

Required environment variables:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
API_SECRET_KEY=your_secret_key_here
```

### 4. Download/Prepare Models

Place your model files in the `models/` directory:
- `yolo_v1_kk_map886.pt` - YOLO detection model
- `unet_kk_cleaner_v1.pt` - U-Net enhancement model (train first if not available)

## Running Locally

### Option 1: Direct Python

```bash
# Run with uvicorn
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Using Python Module

```bash
# Run main.py directly
python src/api/main.py
```

Access the API:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Running with Docker

### Build Image

```bash
docker build -t kk-ocr-v2:latest -f docker/Dockerfile .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e API_SECRET_KEY=your_secret \
  -v $(pwd)/models:/app/models:ro \
  --gpus all \
  kk-ocr-v2:latest
```

### Using Docker Compose

```bash
# Edit docker-compose.yml with your configuration
nano docker/docker-compose.yml

# Start services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

## First API Call

### Using cURL

```bash
curl -X POST "http://localhost:8000/v2/extract/kk" \
  -H "Authorization: Bearer dev-token" \
  -F "file=@path/to/your/kartu_keluarga.jpg"
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v2/extract/kk",
    headers={"Authorization": "Bearer dev-token"},
    files={"file": open("kartu_keluarga.jpg", "rb")}
)

print(response.json())
```

## Training U-Net Model

If you don't have a pre-trained U-Net model:

### 1. Prepare Dataset

Create dataset structure:
```
data/processed/unet_pairs/
├── input/    # Noisy/original crops
│   ├── crop_001.png
│   ├── crop_002.png
│   └── ...
└── target/   # Clean/enhanced crops
    ├── crop_001.png
    ├── crop_002.png
    └── ...
```

### 2. Train Model

```bash
python src/training/train_unet.py \
  --data data/processed/unet_pairs \
  --output models \
  --epochs 100 \
  --batch-size 16 \
  --device cuda
```

### 3. Use Trained Model

The best model will be saved as `models/unet_best.pt`. Update your `.env`:

```bash
MODEL_PATH_UNET=models/unet_best.pt
```

## Monitoring

### Prometheus Metrics

Access metrics at: http://localhost:9090/metrics

### With Docker Compose

If using Docker Compose with Prometheus and Grafana:
- Prometheus: http://localhost:9091
- Grafana: http://localhost:3000 (admin/admin)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## Troubleshooting

### CUDA Not Available

If GPU is not detected:
1. Check NVIDIA drivers: `nvidia-smi`
2. Set CPU device in `.env`:
   ```bash
   YOLO_DEVICE=cpu
   UNET_DEVICE=cpu
   ENABLE_GPU=false
   ```

### Model File Not Found

If you see "model not found" errors:
1. Check model paths in `.env`
2. Ensure model files are in correct location
3. For U-Net, the system will fall back to pass-through mode

### API Authentication Failed

If authentication fails:
1. Check `API_SECRET_KEY` in `.env`
2. For development, use `dev-token` as Bearer token
3. In production, implement proper JWT tokens

### Out of Memory

If you encounter OOM errors:
1. Reduce batch size in `.env`: `UNET_BATCH_SIZE=4`
2. Reduce image size
3. Use CPU instead of GPU

## Next Steps

1. **Production Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md)
2. **Model Training**: See [TRAINING.md](TRAINING.md)
3. **API Documentation**: See [API.md](API.md)
4. **Configuration**: Review `configs/config.yaml`

## Support

For issues:
- Check logs: `logs/kk-ocr.log`
- Enable debug mode: `DEBUG=true` in `.env`
- Review error messages in API response
- Check health endpoint: `curl http://localhost:8000/health`
