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
    
    # MRZ Processing settings
    MRZ_CROP_RATIOS: List[float] = [0.20, 0.25, 0.30, 0.35, 0.40]
    MRZ_UPSCALE_FACTOR: float = 3.0
    
    # National ID / Visual OCR settings
    NATIONAL_ID_CONFIDENCE_THRESHOLD: float = 0.50
    DEFAULT_LANGUAGE_MODE: str = "english_first"
    
    # Tampering Detection settings
    TAMPERING_ENABLED: bool = True
    TAMPERING_ELA_QUALITY: int = 92  # Controlled JPEG recompression quality (90-95)
    TAMPERING_ELA_WEIGHT: float = 0.35
    TAMPERING_NOISE_WEIGHT: float = 0.15
    TAMPERING_EDGE_WEIGHT: float = 0.10
    TAMPERING_COPY_MOVE_WEIGHT: float = 0.15
    TAMPERING_METADATA_WEIGHT: float = 0.10
    TAMPERING_CONSISTENCY_WEIGHT: float = 0.35
    TAMPERING_COMPRESSION_WEIGHT: float = 0.35  # Backward-compatible alias for ELA weight
    TAMPERING_LOW_THRESHOLD: float = 0.30   # 0.00 - 0.29: LOW
    TAMPERING_HIGH_THRESHOLD: float = 0.65  # 0.30 - 0.64: MEDIUM, >= 0.65: HIGH
    TAMPERING_BLOCK_SIZE: int = 32


    TAMPERING_EDITING_SOFTWARE_KEYWORDS: List[str] = [
        "adobe photoshop",
        "photoshop",
        "gimp",
        "gnu image manipulation program",
        "lightroom",
        "affinity photo",
        "paint.net",
        "photopea",
        "coreldraw",
        "pixelmator",
        "canva",
        "indesign",
        "picsart",
        "snapseed",
        "pixlr",
        "seashore"
    ]
    
    # Security
    MASK_PII_LOGS: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")




settings = Settings()
