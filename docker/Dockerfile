# Multi-stage Dockerfile for KK-OCR v2
# Optimized for production deployment with GPU support

# ==================== Base Stage ====================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# ==================== Dependencies Stage ====================
FROM base as dependencies

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ==================== Application Stage ====================
FROM dependencies as application

# Copy application code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY models/ /app/models/

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/logs

# Set working directory
WORKDIR /app

# Expose ports
EXPOSE 8000
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command
CMD ["python3", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]

# ==================== Development Stage (Optional) ====================
FROM application as development

# Install development dependencies
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    pytest-watch

# Set development environment
ENV ENVIRONMENT=development \
    DEBUG=true

CMD ["python3", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload"]
