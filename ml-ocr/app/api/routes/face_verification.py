"""FastAPI route for biometric face verification between identity document and selfie."""

import logging
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, Depends, status

from app.config import settings
from app.models.schemas import FaceVerificationResult, FaceDetail
from app.services.face_verification_service import FaceVerificationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face", tags=["Face Verification"])

ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = settings.FACE_MAX_UPLOAD_SIZE_MB * 1024 * 1024


def get_face_verification_service() -> FaceVerificationService:
    """Dependency provider for FaceVerificationService."""
    return FaceVerificationService()


async def _read_and_validate_image(file: UploadFile, field_name: str) -> np.ndarray:
    """Validates mime type, size, and decodes image in memory without persistent disk storage."""
    if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file.content_type}' for {field_name}. Allowed formats: JPG, JPEG, PNG, WEBP."
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Empty file uploaded for {field_name}."
            )

        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file {field_name} exceeds max allowed size of {settings.FACE_MAX_UPLOAD_SIZE_MB}MB."
            )

        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not decode image data for {field_name}. Ensure it is a valid image file."
            )
        return img
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading {field_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading {field_name}: {str(e)}"
        )


@router.post(
    "/verify",
    response_model=FaceVerificationResult,
    summary="Verify face match between identity document portrait and selfie image",
    description=(
        "Performs biometric face detection, 5-point alignment, and cosine similarity comparison "
        "between an identity document portrait and a live selfie. Returns match decision, "
        "normalized similarity score, and face quality assessments."
    )
)
async def verify_faces(
    document_image: Annotated[UploadFile, File(description="Identity document image containing portrait (JPG/PNG)")],
    selfie_image: Annotated[UploadFile, File(description="Live selfie image containing a single face (JPG/PNG)")],
    verification_service: FaceVerificationService = Depends(get_face_verification_service)
) -> FaceVerificationResult:
    """Compares document portrait against selfie image."""
    doc_img = await _read_and_validate_image(document_image, "document_image")
    selfie_img = await _read_and_validate_image(selfie_image, "selfie_image")

    try:
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(
            verification_service.verify_faces,
            document_image=doc_img,
            selfie_image=selfie_img
        )
        return result

    except Exception as e:
        logger.error(f"Unhandled exception during face verification: {str(e)}", exc_info=True)
        return FaceVerificationResult(
            success=False,
            status="PROCESSING_ERROR",
            match=None,
            similarity_score=None,
            raw_cosine_similarity=None,
            threshold=verification_service.threshold,
            document_face=FaceDetail(detected=False),
            selfie_face=FaceDetail(detected=False),
            warnings=[f"Internal verification error: {str(e)}"],
            disclaimer="Face similarity is a biometric comparison signal and does not by itself prove identity or document authenticity."
        )
