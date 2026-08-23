"""Unified screening route delegating document verification to ScreeningService."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config import settings
from app.models.schemas import DocumentTypeEnum, UnifiedScreeningResult
from app.services.image_service import ImageService
from app.services.screening_service import ScreeningService
from app.services.document_service import DocumentService
from app.services.face_verification_service import FaceVerificationService
from app.services.mrz_service import MRZService
from app.services.ocr_service import OCRService
from app.services.tampering_service import TamperingService
from app.api.routes.ocr import (
    get_ocr_service,
    get_mrz_service,
    get_document_service,
    get_tampering_service,
)
from app.api.routes.face_verification import get_face_verification_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Unified Document Screening"])


def get_screening_service(
    ocr_srv: OCRService = Depends(get_ocr_service),
    mrz_srv: MRZService = Depends(get_mrz_service),
    doc_srv: DocumentService = Depends(get_document_service),
    tampering_srv: TamperingService = Depends(get_tampering_service),
    face_srv: FaceVerificationService = Depends(get_face_verification_service),
) -> ScreeningService:
    """Dependency provider injecting component services into ScreeningService."""
    return ScreeningService(
        ocr_service=ocr_srv,
        mrz_service=mrz_srv,
        document_service=doc_srv,
        tampering_service=tampering_srv,
        face_verification_service=face_srv,
    )


@router.post(
    "/screen",
    response_model=UnifiedScreeningResult,
    status_code=status.HTTP_200_OK,
    summary="Unified identity document screening, tampering forensics, and face verification",
    description=(
        "Accepts a document image (Passport, Visa, National ID) and an optional live selfie image. "
        "Performs complete OCR and MRZ extraction, evaluates multi-signal tampering forensics, "
        "and executes biometric face verification between the document portrait and the selfie."
    ),
)
async def screen_document(
    document_image: UploadFile = File(..., description="Identity document image file (JPG, PNG, WebP, BMP, TIFF)"),
    selfie_image: Optional[UploadFile] = File(None, description="Optional live selfie image file for biometric verification"),
    document_type: str = Form(
        default="auto",
        description="Target document type: 'passport', 'visa', 'national_id', or 'auto' for automatic detection."
    ),
    screening_service: ScreeningService = Depends(get_screening_service),
) -> UnifiedScreeningResult:
    """Processes uploaded document and optional selfie via the unified ScreeningService."""
    logger.info(
        f"Received unified screening request for document: {document_image.filename}, "
        f"selfie: {getattr(selfie_image, 'filename', None)}, document_type: {document_type}"
    )

    # 1. Validate requested document_type parameter
    valid_types = [t.value for t in DocumentTypeEnum]
    if (document_type or "").strip().lower() not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document_type '{document_type}'. Supported values: {', '.join(valid_types)}"
        )

    # 2. Read and validate document image bytes (required)
    try:
        raw_doc_bytes = await ImageService.validate_and_read_upload(document_image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read document image: {str(e)}"
        )

    # 3. Read optional selfie image bytes
    selfie_bytes: Optional[bytes] = None
    if selfie_image is not None and getattr(selfie_image, "filename", None):
        try:
            selfie_bytes = await ImageService.validate_and_read_upload(selfie_image)
        except Exception as e:
            logger.warning(f"Selfie read note: {e}")
            selfie_bytes = None

    # 4. Delegate to ScreeningService off the asyncio event loop
    try:
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(
            screening_service.screen,
            document_bytes=raw_doc_bytes,
            selfie_bytes=selfie_bytes,
            document_type=document_type
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )

    except Exception as exc:
        logger.error(f"Screening pipeline execution error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during unified document screening."
        )
