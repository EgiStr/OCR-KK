#!/bin/bash

# KK-OCR v2 Setup Script
# Automated setup for development environment

set -e  # Exit on error

echo "============================================"
echo "KK-OCR v2 Setup Script"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo -e "${RED}Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓ Pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠ Please edit .env file and set your API keys:${NC}"
    echo "   - GEMINI_API_KEY"
    echo "   - API_SECRET_KEY"
else
    echo -e "${YELLOW}.env file already exists${NC}"
fi

# Check for model files
echo ""
echo "Checking for model files..."
MODELS_MISSING=0

if [ ! -f "models/yolo_v1_kk_map886.pt" ]; then
    echo -e "${YELLOW}⚠ YOLO model not found: models/yolo_v1_kk_map886.pt${NC}"
    MODELS_MISSING=1
fi

if [ ! -f "models/unet_kk_cleaner_v1.pt" ]; then
    echo -e "${YELLOW}⚠ U-Net model not found: models/unet_kk_cleaner_v1.pt${NC}"
    echo "   You can train it using: python src/training/train_unet.py"
    MODELS_MISSING=1
fi

if [ $MODELS_MISSING -eq 0 ]; then
    echo -e "${GREEN}✓ All model files present${NC}"
fi

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p data/raw data/processed data/annotations data/uploads logs
echo -e "${GREEN}✓ Directories created${NC}"

# Run tests (optional)
echo ""
read -p "Run tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running tests..."
    pytest tests/ -v
    echo -e "${GREEN}✓ Tests completed${NC}"
fi

# Summary
echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file and set your API keys"
echo "2. Place model files in models/ directory"
echo "3. Run the application:"
echo "   python -m uvicorn src.api.main:app --reload"
echo ""
echo "Or use Docker:"
echo "   docker-compose -f docker/docker-compose.yml up"
echo ""
echo "For more information, see docs/QUICKSTART.md"
echo ""
