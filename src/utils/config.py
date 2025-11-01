"""
Configuration Management
Load and validate configuration from environment variables
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_SECRET_KEY: str = "change-this-secret-key-in-production"
    API_ALGORITHM: str = "HS256"
    API_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Model Paths
    MODEL_PATH_YOLO: str = "models/yolo_v1_kk_map886.pt"
    MODEL_PATH_UNET: str = "models/unet_kk_cleaner_v1.pt"  # Legacy (for custom trained)
    
    # Enhancement Configuration (Pretrained U-Net)
    USE_PRETRAINED_ENHANCER: bool = True  # Use pretrained (no training needed!)
    ENHANCEMENT_MODEL: str = "Unet"  # Unet, FPN, Linknet, DeepLabV3Plus, etc.
    ENHANCEMENT_ENCODER: str = "resnet34"  # resnet34, efficientnet-b0, mobilenet_v2, etc.
    ENHANCEMENT_METHOD: str = "hybrid"  # hybrid, classical, deep
    
    # VLM Configuration
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-1.5-pro"
    GEMINI_TIMEOUT: int = 900  # seconds
    GEMINI_MAX_RETRIES: int = 3
    
    # YOLO Configuration
    YOLO_CONFIDENCE_THRESHOLD: float = 0.7
    YOLO_INPUT_SIZE: int = 640
    YOLO_DEVICE: str = "cuda"  # cuda or cpu
    
    # U-Net Configuration
    UNET_INPUT_SIZE: int = 256
    UNET_DEVICE: str = "cuda"  # cuda or cpu
    UNET_BATCH_SIZE: int = 8
    
    # Performance Configuration
    MAX_FILE_SIZE_MB: int = 10
    MAX_PROCESSING_TIME_SECONDS: int = 120
    ENABLE_GPU: bool = True
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT: str = "json"  # json or text
    ENABLE_PII_SCRUBBING: bool = True
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Environment
    ENVIRONMENT: str = "development"  # development, staging, production
    DEBUG: bool = False
    
    # Storage (Optional)
    STORAGE_TYPE: str = "local"  # local, s3, gcs
    STORAGE_PATH: str = "data/uploads"
    MODEL_REGISTRY_URL: Optional[str] = None
    DVC_REMOTE: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT == "development"
    
    def get_yolo_device(self) -> str:
        """Get YOLO device (cuda or cpu)"""
        if self.ENABLE_GPU and self.YOLO_DEVICE == "cuda":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"
    
    def get_unet_device(self) -> str:
        """Get U-Net device (cuda or cpu)"""
        if self.ENABLE_GPU and self.UNET_DEVICE == "cuda":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"
    
    @property
    def DEVICE(self) -> str:
        """Get unified device for all models"""
        if self.ENABLE_GPU:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    """
    return Settings()


# Export for convenience
settings = get_settings()
