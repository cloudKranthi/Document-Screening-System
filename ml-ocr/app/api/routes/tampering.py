"""FastAPI route definitions for dedicated document tampering, ELA, and metadata forgery analysis."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.models.schemas import TamperingResult
from app.services.image_service import ImageService
from app.services.tampering_service import TamperingService
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/tampering", tags=["Document Tampering & Forgery Detection"])

# Singleton service instance
tampering_service = TamperingService()


def get_tampering_service() -> TamperingService:
    return tampering_service


@router.post(
    "/analyze",
    response_model=TamperingResult,
    status_code=status.HTTP_200_OK,
    summary="Analyze document image for digital tampering, ELA anomalies & metadata footprints",
    description=(
        "Executes Error Level Analysis (ELA) to detect localized JPEG recompression anomalies, "
        "inspects EXIF container metadata for photo editing signatures (Photoshop, GIMP, etc.), "
        "and returns a normalized tampering risk score (0.0 to 1.0) with explainable evidence."
    )
)
async def analyze_document_tampering_endpoint(
    file: UploadFile = File(..., description="Identity document image file (JPG, JPEG, PNG, WebP, BMP, TIFF)"),
    tampering_srv: TamperingService = Depends(get_tampering_service),
) -> TamperingResult:
    """Primary dedicated endpoint for image tampering and forgery screening."""
    logger.info(f"Received tampering analysis request for file: {file.filename}")
    
    # 1. Safely read and validate upload bytes in memory
    raw_bytes = await ImageService.validate_and_read_upload(file)
    
    # 2. Decode image safely
    original_img, _, _ = ImageService.process_document_image(raw_bytes)

    # 3. Execute tampering evaluation
    result = tampering_srv.analyze_document(
        image_bytes=raw_bytes,
        document_image=original_img
    )
    return result


@router.post(
    "",
    response_model=TamperingResult,
    status_code=status.HTTP_200_OK,
    summary="Analyze document image for digital tampering (alias route)",
    include_in_schema=False
)
async def analyze_document_tampering_alias(
    file: UploadFile = File(..., description="Identity document image file"),
    tampering_srv: TamperingService = Depends(get_tampering_service),
) -> TamperingResult:
    """Alias route for /api/v1/tampering."""
    return await analyze_document_tampering_endpoint(file=file, tampering_srv=tampering_srv)
