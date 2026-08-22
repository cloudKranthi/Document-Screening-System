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


class MRZFormatEnum(str, Enum):
    """Supported ICAO Doc 9303 MRZ formats."""
    TD3 = "TD3"
    MRVA = "MRVA"
    MRVB = "MRVB"
    TD1 = "TD1"
    TD2 = "TD2"


class OCRRegion(BaseModel):
    """Represents an individual OCR detected text region."""
    text: str = Field(..., description="Detected text content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Normalized confidence score (0.0 to 1.0)")
    bbox: List[int] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")


class MRZCheckDigits(BaseModel):
    """Check-digit validation status for individual ICAO 9303 MRZ fields."""
    passport_number: Optional[bool] = Field(None, description="Validity of passport/document number check digit")
    document_number: Optional[bool] = Field(None, description="Validity of document/visa/ID number check digit")
    date_of_birth: Optional[bool] = Field(None, description="Validity of date of birth check digit")
    date_of_expiry: Optional[bool] = Field(None, description="Validity of date of expiry check digit")
    personal_number: Optional[bool] = Field(None, description="Validity of optional/personal number check digit")
    composite: Optional[bool] = Field(None, description="Validity of composite check digit covering all fields")


class MRZCorrection(BaseModel):
    """Details of a deterministic, check-digit verified character correction applied to an MRZ field."""
    line: int = Field(..., description="MRZ line number (1, 2, or 3)")
    position: int = Field(..., description="0-indexed position within the line")
    from_char: str = Field(..., description="Original raw character from OCR")
    to_char: str = Field(..., description="Corrected character")
    field: str = Field(..., description="MRZ field name (e.g. passport_number, date_of_birth, etc.)")
    reason: str = Field(..., description="Explanation of why the substitution was mathematically proven by ICAO check digit")


class FieldValidationItem(BaseModel):
    """Validation result and diagnostics for an individual MRZ/OCR field."""
    valid: bool = Field(..., description="Whether this individual field passes structural and checksum validation")
    value: Optional[str] = Field(None, description="Extracted field value")
    reason: Optional[str] = Field(None, description="Detailed explanation if the field is invalid or uncertain")


class MRZResult(BaseModel):
    """Parsed and validated MRZ information across TD3, MRV-A, MRV-B, TD1, and TD2 formats."""
    detected: bool = Field(..., description="Whether MRZ was detected in the document")
    format: Optional[str] = Field(None, description="Detected MRZ format: TD3, MRVA, MRVB, TD1, TD2, or null")
    line1: Optional[str] = Field(None, description="First normalized/corrected MRZ line")
    line2: Optional[str] = Field(None, description="Second normalized/corrected MRZ line")
    line3: Optional[str] = Field(None, description="Third normalized/corrected MRZ line (for 3-line TD1 format)")
    raw_line1: Optional[str] = Field(None, description="Original uncorrected raw line 1 from OCR")
    raw_line2: Optional[str] = Field(None, description="Original uncorrected raw line 2 from OCR")
    raw_line3: Optional[str] = Field(None, description="Original uncorrected raw line 3 from OCR (for TD1 format)")
    valid_format: bool = Field(..., description="Whether the MRZ adheres to ICAO 9303 line lengths and format")
    check_digits: Optional[MRZCheckDigits] = Field(None, description="Detailed check-digit validation results")
    field_validation: Optional[Dict[str, FieldValidationItem]] = Field(
        default=None,
        description="Detailed field-level validation status and reasons for all extracted MRZ fields"
    )
    overall_valid: bool = Field(..., description="Overall MRZ data consistency check validity (not authenticity)")
    document_code: Optional[str] = Field(None, description="Document type code e.g. P, P<, V, V<, I, A, C")
    issuing_state: Optional[str] = Field(None, description="3-letter ICAO country code of issuing state")
    corrections: List[MRZCorrection] = Field(default_factory=list, description="List of verified character corrections applied")
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


class FieldSourceItem(BaseModel):
    """Metadata regarding field extraction source and confidence."""
    value: Any = Field(..., description="Extracted field value")
    source: str = Field(..., description="Extraction source: 'mrz', 'visual_ocr', or 'hybrid'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score for this field (0.0 to 1.0)")


class OCRExtractResponse(BaseModel):
    """Top-level response model for POST /api/v1/ocr/extract."""
    success: bool = Field(..., description="Indicates if extraction completed successfully")
    document_type: str = Field(..., description="Detected or specified document type (passport, visa, national_id)")
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall average OCR confidence normalized to 0.0-1.0")
    extracted_text: str = Field(..., description="Full consolidated extracted OCR text")
    fields: Dict[str, Any] = Field(default_factory=dict, description="Structured key-value fields extracted from document")
    field_confidences: Optional[Dict[str, float]] = Field(default=None, description="Per-field confidence scores based on underlying OCR detection regions")
    field_sources: Optional[Dict[str, FieldSourceItem]] = Field(default=None, description="Per-field source ('mrz' or 'visual_ocr') and confidence metadata")
    mrz: MRZResult = Field(..., description="MRZ detection, parsed lines, and ICAO 9303 check digit validations")
    field_validation: Optional[Dict[str, FieldValidationItem]] = Field(default=None, description="Field-level validation status and reasons")
    ocr_regions: List[OCRRegion] = Field(default_factory=list, description="Individual OCR detected regions with bounding boxes and confidences")
    processing: ProcessingMetadata = Field(..., description="Image preprocessing and boundary detection metadata")
    language_mode: str = Field(default="english_first", description="Language processing strategy used for visual OCR field extraction")
    warnings: List[str] = Field(default_factory=list, description="Informational warnings or diagnostic notes regarding extraction")
    mrz_debug: Optional[Dict[str, Any]] = Field(None, description="Optional debug metadata containing candidate crops, scores, and best candidate details")
    field_debug: Optional[Dict[str, Any]] = Field(None, description="Optional debug metadata for second-pass field extraction candidates")








class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "1.0.0"
    service: str = "AI-Based Fake Identity & Document Screening - OCR Microservice"
    available_ocr_engines: List[str]
