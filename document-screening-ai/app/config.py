"""Application configuration using Pydantic Settings."""

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment parameters."""
    APP_NAME: str = "AI-Based Fake Identity & Document Screening - OCR Microservice"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Prefix
    API_V1_PREFIX: str = "/api/v1"
    
    # File upload limits
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_MIME_TYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff",
        "application/octet-stream"
    ]
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"]
    
    # Image preprocessing settings
    MAX_IMAGE_DIMENSION: int = 2400
    MIN_IMAGE_DIMENSION: int = 300
    CLAHE_CLIP_LIMIT: float = 2.5
    CLAHE_TILE_GRID_SIZE: int = 8
    
    # OCR Engine settings
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "auto")  # "paddleocr", "tesseract", "auto", "mock"
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "")
    
    # Security
    MASK_PII_LOGS: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
