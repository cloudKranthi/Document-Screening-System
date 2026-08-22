"""Pydantic schemas and response models for the OCR extraction microservice."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentTypeEnum(str, Enum):
    """Supported document types for extraction."""
    PASSPORT = "passport"
    VISA = "visa"
    NATIONAL_ID = "national_id"
    AUTO = "auto"


class OCRRegion(BaseModel):
    """Represents an individual OCR detected text region."""
    text: str = Field(..., description="Detected text content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Normalized confidence score (0.0 to 1.0)")
    bbox: List[int] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")


class MRZCheckDigits(BaseModel):
    """Check-digit validation status for individual ICAO 9303 MRZ fields."""
    passport_number: bool = Field(..., description="Validity of passport number check digit")
    date_of_birth: bool = Field(..., description="Validity of date of birth check digit")
    date_of_expiry: bool = Field(..., description="Validity of date of expiry check digit")
    personal_number: Optional[bool] = Field(None, description="Validity of optional/personal number check digit")
    composite: bool = Field(..., description="Validity of composite check digit covering all fields")


class MRZResult(BaseModel):
    """Parsed and validated MRZ information."""
    detected: bool = Field(..., description="Whether MRZ was detected in the document")
    line1: Optional[str] = Field(None, description="First MRZ line (44 chars for TD3)")
    line2: Optional[str] = Field(None, description="Second MRZ line (44 chars for TD3)")
    valid_format: bool = Field(..., description="Whether the MRZ adheres to ICAO 9303 line lengths and format")
    check_digits: Optional[MRZCheckDigits] = Field(None, description="Detailed check-digit validation results")
    overall_valid: bool = Field(..., description="Overall MRZ data consistency check validity (not authenticity)")
    document_code: Optional[str] = Field(None, description="Document type code e.g. P, P<")
    issuing_state: Optional[str] = Field(None, description="3-letter ICAO country code of issuing state")
    validation_disclaimer: str = Field(
        default="MRZ check-digit validation verifies mathematical data consistency only and does not prove document authenticity.",
        description="Disclaimer regarding authenticity vs consistency"
    )


class ProcessingMetadata(BaseModel):
    """Metadata regarding the image preprocessing and document cropping pipeline."""
    crop_success: bool = Field(..., description="Whether document boundary detection & 4-point perspective crop succeeded")
    preprocessing_applied: bool = Field(..., description="Whether noise reduction, CLAHE, and grayscale enhancements were applied")
    original_dimensions: Optional[List[int]] = Field(None, description="[width, height] of original uploaded image")
    processed_dimensions: Optional[List[int]] = Field(None, description="[width, height] of preprocessed/cropped image")
    boundary_detected: bool = Field(default=False, description="Whether 4-corner document contour was detected")


class OCRExtractResponse(BaseModel):
    """Top-level response model for POST /api/v1/ocr/extract."""
    success: bool = Field(..., description="Indicates if extraction completed successfully")
    document_type: str = Field(..., description="Detected or specified document type (passport, visa, national_id)")
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall average OCR confidence normalized to 0.0-1.0")
    extracted_text: str = Field(..., description="Full consolidated extracted OCR text")
    fields: Dict[str, Any] = Field(default_factory=dict, description="Structured key-value fields extracted from document")
    mrz: MRZResult = Field(..., description="MRZ detection, parsed lines, and ICAO 9303 check digit validations")
    ocr_regions: List[OCRRegion] = Field(default_factory=list, description="Individual OCR detected regions with bounding boxes and confidences")
    processing: ProcessingMetadata = Field(..., description="Image preprocessing and boundary detection metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "1.0.0"
    service: str = "AI-Based Fake Identity & Document Screening - OCR Microservice"
    available_ocr_engines: List[str]
